import os
import time
import uuid
import json
import base64
import io
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory="."), name="images")

device = "cuda"


def image_to_base64(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_8bit",
    quant_kwargs={"load_in_8bit": True}
)

print("Loading model...")
load_start = time.perf_counter()

pipe = DiffusionPipeline.from_pretrained(
    "RedbeardNZ/stable-diffusion-2-1-base",
    dtype=torch.float16,
    quantization_config=pipeline_quant_config,
    safety_checker=None
).to(device)

load_end = time.perf_counter()
print(f"Model loaded in {load_end - load_start:.2f} seconds")

class GenerateRequest(BaseModel):
    prompt: str
    num_steps: int = 20
    num_seeds: int = 1
    num_guidance_samples: int = 1
    guidance_scale_min: float = 7.5
    guidance_scale_max: float = 7.5
    seeds: list[int] | None = None
    width: int = 512
    height: int = 512


def calculate_guidance_scale_values(min_scale: float, max_scale: float, num_samples: int) -> list[float]:
    if num_samples <= 1:
        return [round(min_scale * 2) / 2]

    values: list[float] = []
    step = (max_scale - min_scale) / (num_samples - 1)
    for i in range(num_samples):
        value = min_scale + step * i
        values.append(round(value * 2) / 2)
    return values


@app.post("/generate")
def generate(req: GenerateRequest):
    total_start = time.perf_counter()
    generation_times: list[float] = []
    images_out: list[dict] = []

    # Prepare seeds list
    seeds: list[int] = []
    if req.seeds:
        seeds.extend(req.seeds[: req.num_seeds])

    while len(seeds) < req.num_seeds:
        seeds.append(int(torch.randint(0, 2**31 - 1, (1,)).item()))

    guidance_scales = calculate_guidance_scale_values(
        req.guidance_scale_min,
        req.guidance_scale_max,
        req.num_guidance_samples,
    )

    for seed_index, seed in enumerate(seeds):
        for guidance_index, guidance_scale in enumerate(guidance_scales):
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

            start = time.perf_counter()
            image = pipe(
                req.prompt,
                num_inference_steps=req.num_steps,
                guidance_scale=guidance_scale,
                width=req.width,
                height=req.height,
                generator=generator,
            ).images[0]
            gen_time = time.perf_counter() - start
            generation_times.append(gen_time)

            img_base64 = image_to_base64(image)

            images_out.append(
                {
                    "seed_index": seed_index,
                    "guidance_index": guidance_index,
                    "seed": seed,
                    "guidance_scale": guidance_scale,
                    "imageBase64": img_base64,
                    "generationTime": round(gen_time, 2),
                }
            )

    total_time = time.perf_counter() - total_start
    avg_time = sum(generation_times) / len(generation_times) if generation_times else 0.0

    return {
        "status": "done",
        "images": images_out,
        "totalLatency": round(total_time, 2),
        "averageLatency": round(avg_time, 2),
    }

if __name__ == "__main__":
    uvicorn.run("cloud_server:app", host="0.0.0.0", port=8000, reload=True)
