import os
import sys
import modal
from pathlib import Path

# Define the Modal app
app = modal.App("ltx-2-app")

# Define the image with necessary dependencies
# We use a base image and install dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch>=2.4.0",
        "torchaudio",
        "transformers>=4.49.0",  # 4.52 might not be available yet, let's try latest stable or git.
        # Actually ltx-core said >=4.52. If that doesn't exist on PyPI, we need to install from git.
        # Let's check if 4.52 exists. If not, we might need git+https://github.com/huggingface/transformers.
        # But for now, let's assume the user knows.
        # Wait, if I put >=4.52 and it doesn't exist, pip will fail.
        # Let's try to install from git main because Gemma 3 is very new.
        "git+https://github.com/huggingface/transformers.git",
        "diffusers>=0.30.0",
        "accelerate>=0.33.0",
        "sentencepiece",
        "safetensors",
        "huggingface_hub",
        "opencv-python-headless",
        "einops",
        "imageio",
        "imageio-ffmpeg",
        "numpy",
        "pillow",
        "scipy",
        "av",
        "hf_transfer",
        "fastapi",
        "requests",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Constants
MODEL_ID = "Lightricks/LTX-2"
MODEL_FILENAME = "ltx-2-19b-dev-fp8.safetensors"
GEMMA_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"  # As per README
CACHE_DIR = "/root/model_cache"


def download_model():
    from huggingface_hub import snapshot_download
    import os

    # Ensure HF token is set if available
    if os.environ.get("HF_TOKEN"):
        from huggingface_hub import login

        login(token=os.environ["HF_TOKEN"])

    print(f"Downloading model {MODEL_ID}...")
    snapshot_download(repo_id=MODEL_ID, allow_patterns=[f"*{MODEL_FILENAME}*", "*.json", "*.txt"], local_dir=CACHE_DIR)

    print(f"Downloading Gemma model {GEMMA_ID}...")
    snapshot_download(
        repo_id=GEMMA_ID,
        local_dir=os.path.join(CACHE_DIR, "gemma"),
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Ignore non-essential formats if any
    )

    print("Models downloaded successfully.")


# Run the download function at build time
image = image.run_function(download_model, secrets=[modal.Secret.from_name("huggingface-secret")])

# Add local packages to the image
image = image.add_local_dir(local_path=Path(__file__).parent / "packages", remote_path="/root/packages", copy=True)

# Add packages to PYTHONPATH
image = image.env({"PYTHONPATH": "/root/packages/ltx-core/src:/root/packages/ltx-pipelines/src:$PYTHONPATH"})


@app.cls(
    gpu="H100",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1200,
)
class Model:
    @modal.enter()
    def build(self):
        print("Loading model...")
        import torch
        from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline

        checkpoint_path = Path(CACHE_DIR) / MODEL_FILENAME
        gemma_path = Path(CACHE_DIR) / "gemma"

        self.pipeline = TI2VidOneStagePipeline(
            checkpoint_path=str(checkpoint_path), gemma_root=str(gemma_path), loras=[], device=torch.device("cuda")
        )
        print("Model loaded.")

    def _inference(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 121,
        width: int = 768,
        height: int = 512,
        image_path: str = None,
    ):
        print(f"Generating video for prompt: {prompt}")
        import torch
        from ltx_core.components.guiders import MultiModalGuiderParams

        # Default params
        video_guider_params = MultiModalGuiderParams(
            cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7, modality_scale=1.0, skip_step=0, stg_blocks=[]
        )
        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7, modality_scale=1.0, skip_step=0, stg_blocks=[]
        )

        images = []
        if image_path:
            # (path, frame_index, strength)
            images.append((image_path, 0, 1.0))
            print(f"Using image conditioning from {image_path}")

        with torch.no_grad():
            video, audio = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=42,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=24,
                num_inference_steps=30,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                images=images,
                enhance_prompt=True,
            )

        # Encode video
        from ltx_pipelines.utils.media_io import encode_video

        output_path = "/tmp/output.mp4"
        encode_video(
            video=video, fps=24, audio=audio, audio_sample_rate=48000, output_path=output_path, video_chunks_number=1
        )

        with open(output_path, "rb") as f:
            return f.read()

    @modal.method()
    def generate(
        self, prompt: str, negative_prompt: str = "", num_frames: int = 121, width: int = 768, height: int = 512
    ):
        return self._inference(prompt, negative_prompt, num_frames, width, height)

    @modal.web_endpoint(method="POST")
    def web(self, item: dict):
        from fastapi import Response
        import base64
        import requests
        import uuid

        prompt = item.get("prompt", "A cinematic shot of a futuristic city with flying cars")
        negative_prompt = item.get("negative_prompt", "")
        image_url = item.get("image_url")
        image_base64 = item.get("image_base64")

        image_path = None
        if image_url or image_base64:
            image_path = f"/tmp/{uuid.uuid4()}.png"
            if image_url:
                print(f"Downloading image from {image_url}")
                response = requests.get(image_url)
                with open(image_path, "wb") as f:
                    f.write(response.content)
            elif image_base64:
                print("Decoding base64 image")
                # Remove header if present (e.g., "data:image/png;base64,")
                if "," in image_base64:
                    image_base64 = image_base64.split(",")[1]
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(image_base64))

        try:
            video_bytes = self._inference(prompt, negative_prompt, image_path=image_path)
            return Response(content=video_bytes, media_type="video/mp4")
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)


@app.local_entrypoint()
def main(prompt: str = "A cinematic shot of a futuristic city with flying cars"):
    model = Model()
    video_bytes = model.generate.remote(prompt)
    output_filename = "output.mp4"
    with open(output_filename, "wb") as f:
        f.write(video_bytes)
    print(f"Saved generated video to {output_filename}")
