import os
from fastapi import FastAPI, UploadFile, File # FastAPI para criar a API.
from fastapi.middleware.cors import CORSMiddleware # CORS para liberar acesso da aplicação frontend.
from PIL import Image # PIL (Python Imaging Library) para abrir e processar imagens.
import io
import torch # torch e ultralytics para rodar os modelos YOLO.
from ultralytics import YOLO 
from dotenv import load_dotenv 

load_dotenv() 

# Cria o app FastAPI.
app = FastAPI()

URLS = os.getenv("FRONTEND_URLS")

# Carrega URLs permitidas para CORS (por enquanto “*” permite tudo, isso deve mudar em produção).
app.add_middleware(
    CORSMiddleware, # Configura o CORS para permitir que seu frontend acesse o backend.
    allow_origins=["*"], # altera pra URL depois!!!!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega os dois modelos YOLO previamente treinados (arquivo .pt).
model_binary = YOLO("models/isskin-binary-11l-v1.pt")
model_dx = YOLO("models/isskin-dx-11l-v1.pt")

# Função para redimensionar a imagem para 640x640 pixels, mantendo a proporção e preenchendo com cor preta para manter o tamanho fixo (requisito do YOLO).
def letterbox_image(image, size=(640, 640)): 
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image

# Função para converter bytes da imagem em tensor pronto para o modelo.
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = letterbox_image(image, size=(640, 640))
    return image

# Rota GET /ram que retorna quanta memória RAM está sendo usada (em MB).
@app.get("/ram")
def get_ram_usage():
    import psutil, os
    ram = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return {"ram": round(ram, 2)}

# Rota POST /predict/ que recebe uma imagem, lê ela, roda as duas predições (binary e dx) e retorna os resultados na resposta JSON.
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
