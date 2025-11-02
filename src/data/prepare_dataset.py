import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

RECYCLABLE = {"glass", "paper", "cardboard", "plastic", "metal"}
NON_RECYCLABLE = {"trash"}


def scan_images(raw_dir: str) -> pd.DataFrame:
    rows = []
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            f_l = f.lower()
            if f_l.endswith((".jpg", ".jpeg", ".png")):
                class_name = os.path.basename(root)
                if class_name in RECYCLABLE | NON_RECYCLABLE:
                    path = os.path.join(root, f)
                    label_bin = 1 if class_name in RECYCLABLE else 0
                    rows.append({"path": path.replace("\\", "/"), "class": class_name, "label": label_bin})
    if not rows:
        raise RuntimeError(f"No se encontraron imágenes en {raw_dir}. Asegura la estructura data/raw/dataset-resized/<clase>/*.jpg")
    return pd.DataFrame(rows)


def make_splits(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    # Primero separar test
    trainval_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    # Luego separar val del trainval
    rel_val = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=rel_val,
        random_state=seed,
        stratify=trainval_df["label"],
    )
    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy(); val_df["split"] = "val"
    test_df = test_df.copy(); test_df["split"] = "test"
    return pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="Ruta a data/raw/dataset-resized")
    ap.add_argument("--out_csv", required=True, help="Archivo CSV a generar, ej: data/interim/labels.csv")
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = scan_images(args.raw_dir)
    df = make_splits(df, test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    df.to_csv(args.out_csv, index=False)
    print(f"CSV generado: {args.out_csv} | {len(df)} filas")
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()
