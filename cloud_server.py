import os
import time

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig

app = FastAPI()

# Allow your separate frontend (e.g. http://localhost:3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated images from the project root (where output.png is saved)
app.mount("/images", StaticFiles(directory="."), name="images")

device = "cuda"

# ------------------------------
# Quantization configs (your same settings)
# ------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_8bit",
    quant_kwargs={"load_in_8bit": True}
)

# ------------------------------
# Load model at server startup
# ------------------------------
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

# ------------------------------
# Request schema
# ------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 20


# ------------------------------
# Generate Endpoint
# ------------------------------
@app.post("/generate")
def generate(req: GenerateRequest):
    start = time.perf_counter()

    image = pipe(
        req.prompt,
        num_inference_steps=req.num_inference_steps
    ).images[0]

    output_file = "output.png"
    image.save(output_file)

    total_time = time.perf_counter() - start

    return {
        "status": "done",
        "output_file": output_file,
        "image_url": f"/images/{output_file}",
        "prompt": req.prompt,
        "inference_time": total_time,
    }


if __name__ == "__main__":
    # Run with: python server.py
    # Or directly: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run("cloud_server:app", host="0.0.0.0", port=8000, reload=True)
