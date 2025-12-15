import argparse
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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
    run_name = datetime.now().strftime("cv_%Y%m%d_%H%M%S")
    run_dir = os.path.join(experiments_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    # usar todo menos test si existe
    if (df["split"] == "test").any():
        df = df[df["split"] != "test"].reset_index(drop=True)

    X = df["path"].values
    y = df["label"].values

    folds = int(cfg.get("cv", {}).get("folds", 5))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.get("seed", 42))

    results = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        fold_df = df.copy()
        fold_df.loc[:, "split"] = "train"
        fold_df.loc[va_idx, "split"] = "val"
        fold_csv = os.path.join(run_dir, f"fold_{fold}.csv")
        fold_df.to_csv(fold_csv, index=False)

        # class weights
        tr_labels = fold_df[fold_df["split"] == "train"]["label"].values
        neg = (tr_labels == 0).sum()
        pos = (tr_labels == 1).sum()
        pos_weight = None
        if use_class_weight and pos > 0:
            pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)

        train_loader, val_loader, _ = get_loaders(
            fold_csv, image_size, batch_size, num_workers, aug_strength
        )

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

        for epoch in range(1, num_epochs + 1):
            _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            if val_loss < best_val:
                best_val = val_loss
                best_metrics = val_metrics
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(run_dir, f"best_model_fold{fold}.pt"))
            else:
                no_improve += 1

            if no_improve >= early_stop_patience:
                break

        print(f"Fold {fold}: {best_metrics}")
        results.append({"fold": fold, **best_metrics})

    # aggregate
    metrics_names = list(results[0].keys())
    metrics_names.remove("fold")

    summary = {"per_fold": results}
    for m in metrics_names:
        vals = [r[m] for r in results if r[m] == r[m]]  # ignore NaN
        summary[f"mean_{m}"] = float(np.mean(vals)) if len(vals) else float("nan")
        summary[f"std_{m}"] = float(np.std(vals)) if len(vals) else float("nan")

    with open(os.path.join(run_dir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("CV Summary:", summary)


if __name__ == "__main__":
    main()
