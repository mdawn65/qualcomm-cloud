import os
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig
from huggingface_hub import login
from diffusers import DiffusionPipeline
import time

start_time = time.perf_counter()
device = 'cuda'

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True  # This loads the model weights in int8
)

pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_8bit",
    quant_kwargs={"load_in_8bit": True}
)

# --- Load pipeline with quantization ---
start_load_time = time.perf_counter()
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    dtype=torch.float16,  # fp16 activations
    quantization_config=pipeline_quant_config,  # int8 weights
    safety_checker=None
).to(device)

start_generation_time = time.perf_counter()
prompt = "a boy playing tennis"
image = pipe(prompt, num_inference_steps=50).images[0]
output_file = "output.png"

start_image_save_time = time.perf_counter()
image.save(output_file)

print("Load model time:", start_generation_time - start_load_time)
print("Generate image time:", start_image_save_time - start_generation_time)
print("Save image time:", time.perf_counter() - start_image_save_time)
print("Total time:", time.perf_counter() - start_time)

print(f"Saved image to {output_file}")