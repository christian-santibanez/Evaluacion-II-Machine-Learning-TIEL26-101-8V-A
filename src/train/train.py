import argparse
import os
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.metrics import compute_classification_metrics
from src.data.dataset import get_loaders
from src.models.build_model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    probs_all = []
    y_all = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="val", leave=False):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            probs_all.extend(probs.tolist())
            y_all.extend(y.squeeze(1).detach().cpu().numpy().tolist())
    val_loss = running_loss / len(loader.dataset)
    metrics = compute_classification_metrics(y_all, probs_all)
    return val_loss, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg["training"].get("model_name", "resnet18")
    image_size = int(cfg["training"].get("image_size", 224))
    batch_size = int(cfg["training"].get("batch_size", 32))
    num_epochs = int(cfg["training"].get("num_epochs", 10))
    lr = float(cfg["training"].get("lr", 3e-4))
    weight_decay = float(cfg["training"].get("weight_decay", 1e-4))
    scheduler_name = cfg["training"].get("scheduler", "cosine")
    use_class_weight = bool(cfg["training"].get("class_weight", True))
    num_workers = int(cfg["training"].get("num_workers", 2))
    early_stop_patience = int(cfg["training"].get("early_stopping_patience", 5))
    finetune_strategy = cfg["training"].get("finetune_strategy", "full")
    aug_strength = cfg.get("augmentation", {}).get("strength", "medium")

    csv_path = cfg["paths"]["csv_path"]
    experiments_dir = cfg["paths"]["experiments_dir"]
    os.makedirs(experiments_dir, exist_ok=True)
    run_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    run_dir = os.path.join(experiments_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_loaders(csv_path, image_size, batch_size, num_workers, aug_strength)

    # Compute class weights from train set if enabled
    pos_weight = None
    if use_class_weight:
        import pandas as pd
        df = pd.read_csv(csv_path)
        train_df = df[df["split"] == "train"]
        neg = (train_df["label"] == 0).sum()
        pos = (train_df["label"] == 1).sum()
        if pos > 0:
            pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)

    model, _ = build_model(model_name, finetune_strategy=finetune_strategy)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2)

    best_val = float("inf")
    best_metrics = None
    no_improve = 0

    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    for epoch in range(1, num_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_metrics = val_metrics
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
        else:
            no_improve += 1

        print(f"Epoch {epoch}/{num_epochs} - train_loss: {tr_loss:.4f} val_loss: {val_loss:.4f} F1: {val_metrics['f1_macro']:.4f}")

        if no_improve >= early_stop_patience:
            print("Early stopping")
            break

    # Save history and config
    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(run_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Evaluate on test if available
    if test_loader is not None:
        model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device))
        test_loss, test_metrics = eval_epoch(model, test_loader, criterion, device)
        with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"test_loss": test_loss, **test_metrics}, f, indent=2)
        print("Test:", test_metrics)


if __name__ == "__main__":
    main()
