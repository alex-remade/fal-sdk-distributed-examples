# Distributed Inference Examples

Examples of distributed inference for image generation models using PyTorch distributed computing and the fal platform.

## Overview

This project demonstrates two approaches to running AI image generation models (Stable Diffusion XL, Stable Diffusion 3, PixArt, etc.) across multiple GPUs:

- **Custom Distributed Runner**: ZMQ-based framework for data parallelism (each GPU runs independent model instances)
- **xFuser Integration**: Ray-based framework for model parallelism (model split across GPUs using PipeFusion/Ulysses)

## Installation

### Prerequisites

- Python 3.11+
- CUDA 12.4+
- Multiple GPUs (2-8 recommended)

### Setup

```bash
# Clone and navigate to repository
git clone <repository-url>
cd <repository-directory>

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# For CUDA support
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```

## Project Structure

```
.
├── distributed_example_app/
│   ├── runner/
│   │   ├── demo_app.py                      # Fal app using custom distributed runner
│   │   └── distributed/
│   │       ├── example/
│   │       │   └── distributed_demo_app.py  # SDXL worker implementation
│   │       ├── utils.py                     # Serialization and process utilities
│   │       └── worker.py                    # DistributedWorker and DistributedRunner
│   │
│   └── xfuser/
│       ├── app.py                           # Fal app using xFuser
│       ├── engine.py                        # Ray-based distributed engine
│       └── launch.py                        # Standalone FastAPI server
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

### Key Components

**`distributed_example_app/runner/`**

Custom distributed inference framework using PyTorch distributed and ZMQ:
- `demo_app.py`: Runs Stable Diffusion XL across multiple GPUs, generates image grids
- `distributed/worker.py`: Core classes for distributed execution
  - `DistributedWorker`: Base class for implementing workers
  - `DistributedRunner`: Manages worker processes and ZMQ communication
- `distributed/utils.py`: Serialization and process launching utilities

Use this approach for data parallelism where each GPU runs the full model independently.

**`distributed_example_app/xfuser/`**

xFuser-based distributed inference using advanced parallelism:
- `app.py`: Fal application using xFuser with PipeFusion and Ulysses parallelism
- `engine.py`: Ray-based engine managing multiple xFuser workers
- `launch.py`: Standalone server for local testing

Supported models: SD3 Medium (default), SDXL, PixArt-Alpha/Sigma, HunyuanDiT, FLUX.1

Use this approach for model parallelism where the model is split across GPUs for faster single-image generation.

## Usage

### Local Testing

**Custom Distributed Runner:**
```python
from distributed_example_app.runner.demo_app import ExampleDistributedApp
import fal

app = fal.wrap_app(ExampleDistributedApp)
app()
```

**xFuser:**
```python
from distributed_example_app.xfuser.app import XFuserApp
import fal

app = fal.wrap_app(XFuserApp)
app()
```

### Deploying to Fal

```bash
# Deploy custom distributed runner
fal deploy distributed-demo

# Deploy xFuser app
fal deploy xfuser-demo
```

### Configuration

**Environment Variables:**
- `MODEL_PATH`: HuggingFace model path (default: `stabilityai/stable-diffusion-3-medium-diffusers`)
- `HF_TOKEN`: HuggingFace token for gated models

**GPU Configuration:**
```python
machine_type = "GPU-H100"
num_gpus = 2  # Configurable: 2, 4, or 8
```

**xFuser Parallelism:**
- 2 GPUs: pipefusion=2, ~1.6-1.8x speedup
- 4 GPUs: pipefusion=4, ~2-3x speedup
- 8 GPUs: pipefusion=8, ~3-4x speedup

## API Examples

### Custom Distributed Runner

**Request:**
```json
{
  "prompt": "A fantasy landscape with mountains and rivers",
  "negative_prompt": "blurry, low quality",
  "num_inference_steps": 20,
  "width": 1024,
  "height": 1024
}
```

### xFuser

**Request:**
```json
{
  "prompt": "A serene Japanese garden with cherry blossoms",
  "num_inference_steps": 50,
  "seed": 42,
  "cfg": 7.5,
  "height": 1024,
  "width": 1024
}
```

**Response:**
```json
{
  "image": "https://fal.media/files/...",
  "elapsed_time": "3.45 sec",
  "message": "Image generated successfully"
}
```

## Architecture

### Custom Distributed Runner (Data Parallelism)

```
Fal App → DistributedRunner (Main Process)
              ↓ ZMQ Communication
          GPU0  GPU1  GPU2  GPU3
          (Each runs independent model instance)
```

Main process coordinates work via ZMQ. Each GPU runs a complete model instance. Results are gathered and combined. Good for throughput: process multiple requests in parallel.

### xFuser Engine (Model Parallelism)

```
Fal App → Ray Engine
            ↓ Ray Remote Calls
        xFuser Pipeline
        (model split across GPUs)
        GPU0 | GPU1 | GPU2 | GPU3
```

Ray manages distributed workers. xFuser splits model across GPUs using PipeFusion (pipeline parallelism) and Ulysses (sequence parallelism). Good for latency: generate single images faster.

## Troubleshooting

**Common Issues:**

- "Distributed process group already initialized": Ensure clean shutdown between runs
- CUDA out of memory: Reduce image size or use fewer GPUs
- Timeout errors: Increase timeout configuration
- Model download issues: Set HF_TOKEN for gated models

**Debug Mode:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
