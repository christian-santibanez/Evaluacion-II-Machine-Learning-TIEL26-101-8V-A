import argparse
import os
import json
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.build_model import build_model
from src.utils.config import load_config


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_best_model(exp_dir: str, model_name: str, device: torch.device):
    model, _ = build_model(model_name)
    weights_path = os.path.join(exp_dir, "best_model.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No se encontró best_model.pt en {exp_dir}")
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_image(path: str, model: torch.nn.Module, tfm, device: torch.device) -> float:
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze().item()
    return float(prob)


def find_images_in_dir(directory: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png"}
    paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def save_csv(rows: List[Tuple[str, float, int]], out_path: str):
    dirpath = os.path.dirname(out_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("path,prob_reciclable,pred_label\n")
        for p, prob, pred in rows:
            f.write(f"{p},{prob:.6f},{pred}\n")


def main():
    ap = argparse.ArgumentParser(description="Inferencia con modelo entrenado (binario reciclable vs no reciclable)")
    ap.add_argument("--exp_dir", required=True, help="Directorio del experimento con best_model.pt")
    ap.add_argument("--image", help="Ruta a una imagen para predecir")
    ap.add_argument("--dir", help="Ruta a un directorio con imágenes a predecir (procesa recursivamente)")
    ap.add_argument("--csv_out", help="Ruta para guardar CSV de predicciones (solo con --dir)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Umbral para clasificar (default=0.5)")
    ap.add_argument("--image_size", type=int, default=224, help="Tamaño de entrada del modelo (default=224)")
    ap.add_argument("--config", default="config.yaml", help="Ruta al config base (para leer model_name si no hay config_used.json)")
    args = ap.parse_args()

    if not args.image and not args.dir:
        ap.error("Debe especificar --image o --dir")

    # Determinar model_name desde config_used.json si existe
    used_cfg = os.path.join(args.exp_dir, "config_used.json")
    if os.path.exists(used_cfg):
        cfg = load_config(used_cfg)
    else:
        cfg = load_config(args.config)

    model_name = cfg.get("training", {}).get("model_name", "resnet18")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = get_transform(args.image_size)
    model = load_best_model(args.exp_dir, model_name, device)

    if args.image:
        prob = predict_image(args.image, model, tfm, device)
        pred = 1 if prob >= args.threshold else 0
        label = "reciclable" if pred == 1 else "no_reciclable"
        print(f"Imagen: {args.image}")
        print(f"Prob(reciclable)= {prob:.4f} | Umbral= {args.threshold:.2f} | Pred= {pred} ({label})")
        return

    if args.dir:
        image_paths = find_images_in_dir(args.dir)
        if not image_paths:
            print(f"No se encontraron imágenes en {args.dir}")
            return
        rows = []
        for p in image_paths:
            prob = predict_image(p, model, tfm, device)
            pred = 1 if prob >= args.threshold else 0
            rows.append((p, prob, pred))
        # Resumen en consola
        n = len(rows)
        positives = sum(1 for _, _, pred in rows if pred == 1)
        print(f"Imágenes procesadas: {n} | Predicciones reciclable= {positives} ({positives/n:.1%})")
        if args.csv_out:
            save_csv(rows, args.csv_out)
            print(f"CSV guardado en: {args.csv_out}")


if __name__ == "__main__":
    main()
