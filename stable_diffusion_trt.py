import os
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import logging
import time

# Optional: Reduce logging
logging.set_verbosity_error()

device = "cuda"
start_time = time.perf_counter()

# Load FP16 model (no BitsAndBytes quantization here)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

# --- TensorRT compilation ---
import torch_tensorrt

# Compile UNet
pipe.unet = torch_tensorrt.compile(
    pipe.unet,
    inputs=[torch_tensorrt.Input(
        min_shape=[1, 4, 64, 64],
        opt_shape=[1, 4, 64, 64],
        max_shape=[1, 4, 64, 64],
        dtype=torch.half
    )],
    enabled_precisions={torch.half},  # FP16
    workspace_size=1 << 30
)

# Compile text encoder
pipe.text_encoder = torch_tensorrt.compile(
    pipe.text_encoder,
    inputs=[torch_tensorrt.Input(
        min_shape=[1, 77],
        opt_shape=[1, 77],
        max_shape=[1, 77],
        dtype=torch.half
    )],
    enabled_precisions={torch.half},
    workspace_size=1 << 28
)

load_time = time.perf_counter()
print(f"Model loaded and TensorRT-compiled in {load_time - start_time:.2f}s")

# --- Inference ---
prompt = "a boy playing tennis"
start_gen_time = time.perf_counter()
image = pipe(prompt, num_inference_steps=50).images[0]
end_gen_time = time.perf_counter()

# Save image
output_file = "output.png"
start_save_time = time.perf_counter()
image.save(output_file)
end_save_time = time.perf_counter()

print("Generate image time:", end_gen_time - start_gen_time)
print("Save image time:", end_save_time - start_save_time)
print("Total time:", end_save_time - start_time)
print(f"Saved image to {output_file}")
