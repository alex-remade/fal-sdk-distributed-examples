import asyncio
import base64
import io
import time
from typing import Any, Optional

import fal
from fal.toolkit import File, Image, clone_repository
from pydantic import BaseModel, Field


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


class XFuserApp(
    fal.App,
    keep_alive=300,

):
    """
    Fal app that runs xFuser for distributed image generation.
    
    This app uses Ray to distribute inference across multiple GPUs.
    
    Supported Models (set via MODEL_PATH environment variable):
    - stabilityai/stable-diffusion-3-medium-diffusers (SD3 - default, DiT architecture)
    - PixArt-alpha/PixArt-Sigma-XL-2-1024-MS (PixArt-Sigma 1024 - DiT)
    - PixArt-alpha/PixArt-Sigma-XL-2-2K-MS (PixArt-Sigma 2K - DiT)
    - PixArt-alpha/PixArt-XL-2-1024-MS (PixArt-Alpha - DiT)
    - Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers (HunyuanDiT - DiT)
    - stabilityai/stable-diffusion-xl-base-1.0 (SDXL - U-Net, requires CFG parallel with 2 GPUs)
    
    Configuration via environment variables:
    - MODEL_PATH: HuggingFace model path (default: stabilityai/stable-diffusion-3-medium-diffusers)
    - HF_TOKEN: HuggingFace token for authenticated model access (required for SD3)
    - PIPEFUSION_DEGREE: Pipeline fusion parallelism degree (default: 2)
    - ULYSSES_DEGREE: Ulysses sequence parallelism degree (default: 1)
    - RING_DEGREE: Ring attention parallelism degree (default: 1)
    - WARMUP_STEPS: Number of warmup steps (default: 1)
    
    Note: CFG parallel is automatically enabled for U-Net models (SDXL) when using 2 GPUs.
          For DiT models, dit_parallel_size is automatically calculated.
    """

    num_gpus = 2
    
    requirements = [
        "torch>=2.6.0",
        "torchvision>=0.21.0",
        "diffusers>=0.28.2",
        "transformers>=4.47.2,<4.52.0",
        "accelerate>=1.4.0",
        "pyzmq>=25.0.0",
        "ray>=2.0.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.9.0.80",
        "fastapi>=0.100.0",
        "httpx>=0.24.0",
        "pydantic>=1.8,<2.0",  # Locked by Fal platform
        "uvicorn>=0.20.0",
        "xfuser>=0.3.0",
        "sentencepiece",
        "protobuf",
    ]
    
    machine_type="GPU-H100"


    async def setup(self) -> None:
        """
        Initialize the xFuser distributed engine.
        """
        import os
        import sys
        import ray

        print("=== Starting xFuser Distributed Engine Setup ===")

        # Clone the repository containing distributed_example_app
        print("Cloning repository...")
        repo_path = clone_repository(
            "https://github.com/alex-remade/fal-sdk-distributed-examples",
            include_to_path=True,
            commit_hash="a30d2f91cf47f59a215ad145cb57489a4123b213"
        )
        
        print(f"Repository cloned to: {repo_path}")
        
        # Explicitly ensure it's in sys.path
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
            print(f"Added {repo_path} to sys.path")
        
        os.chdir(repo_path)

        # Import from cloned repository
        print("Attempting to import distributed_example_app.engine...")
        from distributed_example_app.engine import Engine
        from xfuser import xFuserArgs
        print("Import successful!")

        # Set HuggingFace token if provided
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            print("✓ HuggingFace token configured")
        else:
            print("⚠ WARNING: No HF_TOKEN found - may fail for gated models")

        # Load configuration from environment variables
        # Default to SD3 Medium (DiT model with excellent distributed support)
        model_path = os.environ.get("MODEL_PATH", "stabilityai/stable-diffusion-3-medium-diffusers")
        world_size = self.num_gpus
        
        # Parallelism configuration
        pipefusion_degree = 2
        ulysses_degree = 1
        ring_degree = 1
        warmup_steps = 1
        use_cfg_parallel = False

        dit_parallel_size = pipefusion_degree * ulysses_degree * (2 if use_cfg_parallel else 1)


        # Log configuration
        print("=== xFuser Configuration ===")
        print(f"Model: {model_path}")

        print(f"World Size: {world_size}")
        print(f"PipeFusion Degree: {pipefusion_degree}")
        print(f"Ulysses Degree: {ulysses_degree}")
        print(f"Ring Degree: {ring_degree}")
        print(f"Use CFG Parallel: {use_cfg_parallel}")
        print(f"DiT Parallel Size: {dit_parallel_size}")
        print(f"Warmup Steps: {warmup_steps}")
        print("============================")

        # Create xFuser configuration as a dict (avoids Pydantic pickling issues)
        xfuser_args_dict = {
            'model': model_path,
            'trust_remote_code': True,
            'warmup_steps': warmup_steps,
            'use_parallel_vae': False,
            'use_torch_compile': False,
            'ulysses_degree': ulysses_degree,
            'pipefusion_parallel_degree': pipefusion_degree,
            'use_cfg_parallel': use_cfg_parallel,
            'dit_parallel_size': dit_parallel_size,
        }

        # Initialize the distributed engine
        print("Initializing Ray engine...")
        self.engine = Engine(
            world_size=world_size,
            xfuser_args_dict=xfuser_args_dict
        )
        print("Ray engine initialized successfully!")

        # Optional: Run warmup
        if warmup_steps > 0:
            print("Running warmup generation...")
            try:
                warmup_request = {
                    "prompt": "a cat wearing a hat",
                    "num_inference_steps": 20,
                    "height": 512,
                    "width": 512,
                    "seed": 42,
                    "cfg": 7.5,
                }
                warmup_result = self.engine.generate(warmup_request)
                print(f"Warmup completed: {warmup_result.get('message', 'success')}")
            except Exception as e:
                print(f"Warning: Warmup failed: {e}")

        print("=== xFuser Setup Complete ===")

    @fal.endpoint("/")
    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse:
        """
        Generate an image using xFuser distributed inference.
        
        This endpoint uses the Ray-based distributed engine to generate images.
        
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
        
        # Convert Pydantic model to dict for Ray compatibility
        request_dict = request.dict()
        
        # Call the engine directly (synchronous Ray call)
        result = self.engine.generate(request_dict)

        # Process the result
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
        import ray
        
        print("Cleaning up xFuser engine...")
        
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()
            print("Ray shutdown complete")


if __name__ == "__main__":
    app = fal.wrap_app(XFuserApp)
    app()
