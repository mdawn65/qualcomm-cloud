import os
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import logging
import time
import torch_tensorrt

logging.set_verbosity_error()  # suppress HF warnings

device = "cuda"
model_name = "RedbeardNZ/stable-diffusion-2-1-base"
output_file = "output_trt.png"
prompt = "a boy playing tennis"
num_inference_steps = 20

start_time = time.perf_counter()

# --- Load pipeline in FP16 ---
pipe = DiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

# --- TensorRT conversion ---
print("Converting UNet and VAE to TensorRT...")

# Convert UNet
unet_trt = torch_tensorrt.ts.convert_module(
    pipe.unet,
    inputs=[torch_tensorrt.ts.Input(torch.randn(1, 4, 64, 64).to(device), dtype=torch.half)],
    enabled_precisions={torch.half},  # use FP16
    workspace_size=1 << 30  # 1GB workspace
)
pipe.unet = unet_trt

# Convert VAE decoder
vae_trt = torch_tensorrt.ts.convert_module(
    pipe.vae.decoder,
    inputs=[torch_tensorrt.ts.Input(torch.randn(1, 4, 64, 64).to(device), dtype=torch.half)],
    enabled_precisions={torch.half},
    workspace_size=1 << 30
)
pipe.vae.decoder = vae_trt

# --- Generate image ---
start_gen_time = time.perf_counter()
image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]

# --- Save image ---
image.save(output_file)

print("Total time:", time.perf_counter() - start_time)
print("Saved image to", output_file)
