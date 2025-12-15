import io
import os
from typing import Tuple

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import onnxruntime as ort
import torch
from torchvision import transforms

from src.utils.config import load_config


def load_onnx_session(exp_dir: str) -> Tuple[ort.InferenceSession, int]:
    """Carga el modelo ONNX y devuelve la sesión y el tamaño de imagen.

    Se asume que en exp_dir existen:
      - model.onnx
      - config_used.json (o en su defecto se usará config.yaml en la raíz)
    """
    onnx_path = os.path.join(exp_dir, "model.onnx")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"No se encontró model.onnx en {exp_dir}")

    used_cfg_path = os.path.join(exp_dir, "config_used.json")
    if os.path.isfile(used_cfg_path):
        cfg = load_config(used_cfg_path)
    else:
        # fallback: config.yaml en la raíz del proyecto
        cfg = load_config("config.yaml")

    image_size = int(cfg["training"].get("image_size", 224))

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    return session, image_size


def build_preprocess(image_size: int):
    # Debe ser coherente con el preprocesamiento usado en entrenamiento
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


EXP_DIR = os.getenv("TRASH_EXP_DIR", "experiments/exp_20251215_014000")

session, _IMAGE_SIZE = load_onnx_session(EXP_DIR)
preprocess = build_preprocess(_IMAGE_SIZE)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

app = FastAPI(
    title="Clasificación de Residuos - FastAPI + ONNX",
    description=(
        "Servicio de inferencia para el modelo de clasificación binaria de residuos "
        "entrenado con Deep Learning (ResNet18) y exportado a ONNX. "
        "Permite subir una imagen y obtener la predicción (0 = No Reciclable, 1 = Reciclable)."
    ),
    version="1.0.0",
)


def _prepare_image(file_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo leer la imagen: {e}")

    x = preprocess(img)  # [C, H, W] tensor
    x = x.unsqueeze(0)  # [1, C, H, W]
    return x.numpy().astype(np.float32)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Recibe una imagen y devuelve la clase predicha y la probabilidad.

    - label: 0 = No Reciclable, 1 = Reciclable
    - prob: probabilidad (sigmoid) de ser clase 1
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen")

    file_bytes = await file.read()
    input_array = _prepare_image(file_bytes)

    # Ejecutar inferencia ONNX
    outputs = session.run([output_name], {input_name: input_array})
    logits = outputs[0]  # [1, 1] o [1]
    # Aplicar sigmoid para obtener probabilidad
    logits_tensor = torch.from_numpy(logits).float()
    prob = torch.sigmoid(logits_tensor).item()
    label = int(prob >= 0.5)

    return JSONResponse(
        {
            "label": label,
            "prob": prob,
            "description": "1 = Reciclable, 0 = No reciclable",
            "experiment_dir": EXP_DIR,
        }
    )


@app.get("/")
async def root():
    return {
        "message": "Servicio de clasificación de residuos activo. Ir a /docs para probar el endpoint /predict.",
        "experiment_dir": EXP_DIR,
    }
