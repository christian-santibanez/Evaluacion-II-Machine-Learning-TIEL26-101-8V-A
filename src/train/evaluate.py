import argparse
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataset import TrashBinaryDataset
from src.models.build_model import build_model


def load_history(exp_dir: str):
    path = os.path.join(exp_dir, "history.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def plot_history(exp_dir: str, out_dir: str):
    hist = load_history(exp_dir)
    if not hist:
        return None

    os.makedirs(out_dir, exist_ok=True)

    # Plot train/val loss
    plt.figure(figsize=(7, 4))
    plt.plot(hist.get("train_loss", []), label="train_loss")
    plt.plot(hist.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    loss_path = os.path.join(out_dir, "loss_curves.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    # Plot val F1 if available
    val_metrics = hist.get("val_metrics", [])
    if val_metrics:
        f1s = [m.get("f1_macro", np.nan) for m in val_metrics]
        accs = [m.get("accuracy", np.nan) for m in val_metrics]
        plt.figure(figsize=(7, 4))
        plt.plot(f1s, label="val_f1_macro")
        plt.plot(accs, label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Metrics")
        plt.legend()
        vm_path = os.path.join(out_dir, "val_metrics_curves.png")
        plt.tight_layout()
        plt.savefig(vm_path, dpi=150)
        plt.close()

    return True


def get_test_loader(csv_path: str, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    df = pd.read_csv(csv_path)
    if not (df["split"] == "test").any():
        return None
    ds = TrashBinaryDataset(csv_path, split="test", image_size=image_size, aug_strength="none")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def evaluate_test(exp_dir: str, cfg_path: str, out_dir: str):
    # Load config used
    used_cfg_path = os.path.join(exp_dir, "config_used.json")
    cfg = load_config(used_cfg_path if os.path.exists(used_cfg_path) else cfg_path)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg["training"].get("model_name", "resnet18")
    image_size = int(cfg["training"].get("image_size", 224))
    batch_size = int(cfg["training"].get("batch_size", 32))
    num_workers = int(cfg["training"].get("num_workers", 2))

    csv_path = cfg["paths"]["csv_path"]

    loader = get_test_loader(csv_path, image_size, batch_size, num_workers)
    if loader is None:
        print("No existe split de test en el CSV. Saltando evaluación de test.")
        return None

    model, _ = build_model(model_name)
    best_weights = os.path.join(exp_dir, "best_model.pt")
    model.load_state_dict(torch.load(best_weights, map_location=device))
    model.to(device)
    model.eval()

    probs_all = []
    y_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            probs_all.extend(probs.tolist())
            y_all.extend(y.numpy().tolist())

    y_true = np.asarray(y_all).astype(int)
    y_prob = np.asarray(probs_all)
    y_pred = (y_prob >= 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Reciclable (0)", "Reciclable (1)"],
                yticklabels=["No Reciclable (0)", "Reciclable (1)"])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión (Test)")
    cm_path = os.path.join(out_dir, "confusion_matrix_test.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.legend(loc="lower right")
    roc_path = os.path.join(out_dir, "roc_curve_test.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    return {
        "confusion_matrix_path": cm_path,
        "roc_curve_path": roc_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="Directorio del experimento (experiments/exp_YYYYMMDD_HHMMSS)")
    ap.add_argument("--config", default="config.yaml", help="Ruta al config base")
    ap.add_argument("--out_dir", default="report/figuras", help="Directorio de salida para figuras")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Curvas de entrenamiento/validación
    plot_history(args.exp_dir, args.out_dir)

    # 2) Evaluación en test con figuras
    evaluate_test(args.exp_dir, args.config, args.out_dir)

    print(f"Figuras generadas en: {args.out_dir}")


if __name__ == "__main__":
    main()
