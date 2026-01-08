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
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: int | None = None

@app.post("/generate")
def generate(req: GenerateRequest):
    def event_stream():
        total_start = time.perf_counter()
        generation_times: list[float] = []

        seed_index = 0
        guidance_index = 0

        generator = torch.Generator(device=device)
        if req.seed is not None:
            generator.manual_seed(req.seed)

        start = time.perf_counter()
        image = pipe(
            req.prompt,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
            generator=generator,
        ).images[0]
        gen_time = time.perf_counter() - start
        generation_times.append(gen_time)

        img_base64 = image_to_base64(image)

        image_event = {
            "type": "image",
            "seed_index": seed_index,
            "guidance_index": guidance_index,
            "seed": req.seed,
            "guidance_scale": req.guidance_scale,
            "imageBase64": img_base64,
            "generationTime": round(gen_time, 2),
            "clipScore": None,
            "clipComputationTime": None,
        }
        yield "data: " + json.dumps(image_event) + "\n\n"

        total_time = time.perf_counter() - total_start
        avg_time = sum(generation_times) / len(generation_times) if generation_times else 0.0

        complete_event = {
            "type": "complete",
            "totalLatency": round(total_time, 2),
            "averageLatency": round(avg_time, 2),
        }
        yield "data: " + json.dumps(complete_event) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("cloud_server:app", host="0.0.0.0", port=8000, reload=True)
