"""
Configuration file for xFuser fal app.

Set environment variables before deploying:
    export MODEL_PATH="black-forest-labs/FLUX.1-schnell"
    export PIPEFUSION_DEGREE="2"
    export ULYSSES_DEGREE="1"
"""

# Model configurations
MODELS = {
    "flux-schnell": {
        "path": "black-forest-labs/FLUX.1-schnell",
        "default_steps": 4,
        "machine_type": "GPU-A100",
        "num_gpus": 2,
        "pipefusion_degree": 2,
        "ulysses_degree": 1,
    },
    "flux-dev": {
        "path": "black-forest-labs/FLUX.1-dev",
        "default_steps": 20,
        "machine_type": "GPU-A100",
        "num_gpus": 2,
        "pipefusion_degree": 2,
        "ulysses_degree": 1,
    },
    "sd3": {
        "path": "stabilityai/stable-diffusion-3-medium-diffusers",
        "default_steps": 28,
        "machine_type": "GPU-A100",
        "num_gpus": 2,
        "pipefusion_degree": 2,
        "ulysses_degree": 1,
    },
    "pixart-sigma": {
        "path": "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS",
        "default_steps": 20,
        "machine_type": "GPU-A100",
        "num_gpus": 2,
        "pipefusion_degree": 2,
        "ulysses_degree": 1,
    },
}

