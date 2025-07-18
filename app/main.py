import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import psutil
import torch
from torchvision import transforms
from ultralytics import YOLO 
from dotenv import load_dotenv

load_dotenv() 

app = FastAPI()

URLS = os.getenv("FRONTEND_URLS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # altera pra URL depois!!!!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load both models
model_binary = YOLO("models/isskin-binary-11l-v1.pt")
model_dx = YOLO("models/isskin-dx-11l-v1.pt")

def letterbox_image(image, size=(640, 640)): 
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = letterbox_image(image, size=(640, 640))
    return image

@app.get("/ram")
def get_ram_usage():
    import psutil, os
    ram = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return {"ram": round(ram, 2)}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_image = preprocess_image(image_bytes)

    with torch.no_grad():
        # Run binary model
        result_binary = model_binary(input_image, conf=0.25, iou=0.45)[0]
        probs_binary = result_binary.probs.data.tolist()
        class_names_binary = result_binary.names
        image_path_binary = result_binary.path
        timing_binary = {k: round(v / 1000, 4) for k, v in result_binary.speed.items()}

        # Run diagnosis model
        result_dx = model_dx(input_image, conf=0.25, iou=0.45)[0]
        probs_dx = result_dx.probs.data.tolist()
        class_names_dx = result_dx.names
        image_path_dx = result_dx.path
        timing_dx = {k: round(v / 1000, 4) for k, v in result_dx.speed.items()}

    prediction_binary = {
        class_names_binary[i]: round(prob * 100, 2)
        for i, prob in enumerate(probs_binary)
    }

    prediction_dx = {
        class_names_dx[i]: round(prob * 100, 2)
        for i, prob in enumerate(probs_dx)
    }

    return {
        "binary_prediction": prediction_binary,
        "dx_prediction": prediction_dx,
        "metadata": {
            "image_path_binary": image_path_binary,
            "image_path_dx": image_path_dx,
            "processing_time_binary": timing_binary,
            "processing_time_dx": timing_dx
        }
    }
