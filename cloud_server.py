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
from transformers import BitsAndBytesConfig, CLIPProcessor, CLIPModel
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

clip_model: CLIPModel | None = None
clip_processor: CLIPProcessor | None = None


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

print("Loading CLIP model for scoring...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP model loaded successfully!")


def compute_clip_score_fn(pil_image: Image.Image, text: str) -> tuple[float, float]:
    start_time = time.time()

    inputs = clip_processor(
        text=[text],
        images=pil_image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
    text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    clip_score = (image_features @ text_features.T).item()
    return clip_score, time.time() - start_time

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
    compute_clip_score: bool = False


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
    def event_stream():
        total_start = time.perf_counter()
        generation_times: list[float] = []

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

                clip_score = None
                clip_time = None
                if req.compute_clip_score:
                    clip_score, clip_time = compute_clip_score_fn(image, req.prompt)

                img_base64 = image_to_base64(image)

                image_event = {
                    "type": "image",
                    "seed_index": seed_index,
                    "guidance_index": guidance_index,
                    "seed": seed,
                    "guidance_scale": guidance_scale,
                    "imageBase64": img_base64,
                    "generationTime": round(gen_time, 2),
                    "clipScore": round(clip_score, 4) if clip_score is not None else None,
                    "clipComputationTime": round(clip_time, 2) if clip_time is not None else None,
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
