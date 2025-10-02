import asyncio
import base64
import io
import time
from typing import Any, Optional, TYPE_CHECKING

import fal
from fal.container import ContainerImage
from fal.toolkit import File, Image
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import httpx
    from fastapi.responses import Response

# Define the Docker container for xFuser
dockerfile_str = """
FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    xfuser>=0.3.0 \
    ray \
    httpx \
    diffusers \
    transformers \
    accelerate \
    sentencepiece \
    protobuf \
    fastapi \
    uvicorn \
    pydantic

WORKDIR /app
"""


class GenerateRequest(BaseModel):
    """Request model for image generation using xFuser."""

    prompt: str = Field(description="Text prompt for image generation")
    num_inference_steps: int = Field(default=50, description="Number of inference steps")
    seed: int = Field(default=42, description="Random seed for generation")
    cfg: float = Field(default=7.5, description="Classifier-free guidance scale")
    height: int = Field(default=1024, description="Image height")
    width: int = Field(default=1024, description="Image width")
    save_disk_path: Optional[str] = Field(
        default=None,
        description="Optional path to save the image to disk instead of returning base64",
    )


class GenerateResponse(BaseModel):
    """Response model containing the generated image."""

    image: File = Field(description="Generated image")
    elapsed_time: str = Field(description="Time taken to generate the image")
    message: str = Field(description="Status message")


class XFuserApp(fal.App):
    """
    Fal app that runs xFuser for distributed image generation.
    
    This app starts the xFuser service as a subprocess and forwards
    requests to its internal API.
    
    Configuration via environment variables:
    - MODEL_PATH: HuggingFace model path (default: black-forest-labs/FLUX.1-schnell)
    - PIPEFUSION_DEGREE: Pipeline fusion parallelism degree (default: 2)
    - ULYSSES_DEGREE: Ulysses sequence parallelism degree (default: 1)
    - RING_DEGREE: Ring attention parallelism degree (default: 1)
    - USE_CFG_PARALLEL: Enable CFG parallelism (default: false)
    - SAVE_DISK_PATH: Default path for saving images (default: output)
    - WARMUP_STEPS: Number of warmup steps (default: 1)
    """

    machine_type = "GPU-A100"
    num_gpus = 2
    kind = "container"
    container_image = ContainerImage.from_dockerfile_str(dockerfile_str)
    local_files = [
        "launch.py",
    ]
    keep_alive = 300

    async def setup(self) -> None:
        """
        Start the xFuser service as a subprocess and wait for it to be ready.
        """
        import os
        import httpx

        # Initialize instance variables (don't use __init__)
        self.process: Optional[asyncio.subprocess.Process] = None
        self.internal_api_url = "http://127.0.0.1:6000"
        self.client: Optional[httpx.AsyncClient] = None

        # Load configuration from environment variables
        model_path = os.environ.get("MODEL_PATH", "black-forest-labs/FLUX.1-schnell")
        world_size = self.num_gpus
        
        # Parallelism configuration
        pipefusion_degree = int(os.environ.get("PIPEFUSION_DEGREE", "2"))
        ulysses_degree = int(os.environ.get("ULYSSES_DEGREE", "1"))
        ring_degree = int(os.environ.get("RING_DEGREE", "1"))
        
        # Additional options
        save_disk_path = os.environ.get("SAVE_DISK_PATH", "output")
        use_cfg_parallel = os.environ.get("USE_CFG_PARALLEL", "false").lower() == "true"
        
        # Advanced options
        warmup_steps = int(os.environ.get("WARMUP_STEPS", "1"))

        # Build command to start the xFuser service
        cmd = [
            "python",
            "launch.py",
            "--model_path",
            model_path,
            "--world_size",
            str(world_size),
            "--pipefusion_parallel_degree",
            str(pipefusion_degree),
            "--ulysses_parallel_degree",
            str(ulysses_degree),
            "--ring_degree",
            str(ring_degree),
            "--save_disk_path",
            save_disk_path,
        ]
        
        if use_cfg_parallel:
            cmd.append("--use_cfg_parallel")

        # Log configuration
        print("=== xFuser Configuration ===")
        print(f"Model: {model_path}")
        print(f"World Size: {world_size}")
        print(f"PipeFusion Degree: {pipefusion_degree}")
        print(f"Ulysses Degree: {ulysses_degree}")
        print(f"Ring Degree: {ring_degree}")
        print(f"Use CFG Parallel: {use_cfg_parallel}")
        print(f"Warmup Steps: {warmup_steps}")
        print("============================")

        # Start the subprocess
        print(f"Starting xFuser service with command: {' '.join(cmd)}")
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Start background tasks to log subprocess output
        async def log_output(stream, prefix):
            while True:
                line = await stream.readline()
                if not line:
                    break
                print(f"[{prefix}] {line.decode().rstrip()}")

        asyncio.create_task(log_output(self.process.stdout, "xFuser-OUT"))
        asyncio.create_task(log_output(self.process.stderr, "xFuser-ERR"))

        # Create HTTP client
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout

        # Wait for the service to be ready
        print("Waiting for xFuser service to be ready...")
        max_retries = 60
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                response = await self.client.get(f"{self.internal_api_url}/health")
                if response.status_code == 200:
                    print("xFuser service is ready!")
                    break
                else:
                    print(f"Attempt {attempt + 1}/{max_retries}: Got status {response.status_code}, retrying...")
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries}: Connection failed ({type(e).__name__}: {e}), retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    raise RuntimeError(f"Failed to start xFuser service after {max_retries} attempts: {e}")

        # Warmup request
        print("Running warmup request...")
        try:
            warmup_request = GenerateRequest(
                prompt="a cat wearing a hat",
                num_inference_steps=20,
                height=512,
                width=512,
            )
            warmup_result = await self._forward_request(warmup_request)
            print(f"Warmup completed: {warmup_result.get('message', 'success')}")
        except Exception as e:
            print(f"Warning: Warmup failed: {e}")

    async def _forward_request(self, request: GenerateRequest) -> dict[str, Any]:
        """
        Forward a request to the internal xFuser API.
        """
        if not self.client:
            raise RuntimeError("HTTP client not initialized")

        response = await self.client.post(
            f"{self.internal_api_url}/generate",
            json=request.dict(),
        )

        if response.status_code != 200:
            raise RuntimeError(f"xFuser service returned error: {response.text}")

        return response.json()

    @fal.endpoint("/")
    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse:
        """
        Generate an image using xFuser distributed inference.
        
        This endpoint forwards the request to the internal xFuser service
        and returns the generated image.
        
        Parameters:
        - prompt: Text description of the image to generate
        - num_inference_steps: Number of denoising steps (more = higher quality but slower)
        - seed: Random seed for reproducibility
        - cfg: Classifier-free guidance scale (higher = more prompt adherence)
        - height: Image height in pixels
        - width: Image width in pixels
        - save_disk_path: Optional path to save to disk (mostly for debugging)
        """
        start_time = time.time()
        
        # Forward request to internal API
        result = await self._forward_request(request)

        # Check if we got a base64 image
        if not result.get("save_to_disk", False):
            # Decode base64 image
            img_data = base64.b64decode(result["output"])
            img_bytes = io.BytesIO(img_data)
            
            from PIL import Image as PILImage
            pil_image = PILImage.open(img_bytes)
            
            return GenerateResponse(
                image=Image.from_pil(pil_image),
                elapsed_time=result.get("elapsed_time", f"{time.time() - start_time:.2f} sec"),
                message=result.get("message", "Image generated successfully"),
            )
        else:
            # Handle file path response (if save_disk_path was specified)
            file_path = result["output"]
            from PIL import Image as PILImage
            pil_image = PILImage.open(file_path)
            
            return GenerateResponse(
                image=Image.from_pil(pil_image),
                elapsed_time=result.get("elapsed_time", f"{time.time() - start_time:.2f} sec"),
                message=result.get("message", "Image generated successfully"),
            )

    async def cleanup(self) -> None:
        """
        Clean up resources when shutting down.
        """
        if self.client:
            await self.client.aclose()

        if self.process:
            print("Shutting down xFuser service...")
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                print("Force killing xFuser service...")
                self.process.kill()
                await self.process.wait()
    
    


if __name__ == "__main__":
    app = fal.wrap_app(XFuserApp)
    app()

