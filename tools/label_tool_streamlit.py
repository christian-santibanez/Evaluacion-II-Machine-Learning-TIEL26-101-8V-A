import argparse
import os
import pandas as pd
import streamlit as st


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Ruta al CSV de etiquetas (data/interim/labels.csv)")
    args, _ = ap.parse_known_args()
    return args


def main():
    args = parse_args()
    st.set_page_config(page_title="Herramienta de Etiquetado - Reciclable vs No", layout="wide")
    st.title("Herramienta de Etiquetado - Reciclable vs No Reciclable")

    if not os.path.exists(args.csv):
        st.error(f"No existe el CSV: {args.csv}")
        st.stop()

    if "df" not in st.session_state:
        df = pd.read_csv(args.csv)
        # asegurar columnas
        expected_cols = {"path", "class", "label", "split"}
        missing = expected_cols - set(df.columns)
        if missing:
            st.error(f"Faltan columnas en CSV: {missing}")
            st.stop()
        st.session_state.df = df
        st.session_state.idx = 0
        st.session_state.split = "train"

    df = st.session_state.df

    # Filtros
    splits = sorted(df["split"].unique().tolist())
    st.session_state.split = st.sidebar.selectbox("Split", splits, index=splits.index(st.session_state.split) if st.session_state.split in splits else 0)
    sub = df[df["split"] == st.session_state.split].reset_index(drop=True)

    if len(sub) == 0:
        st.info("No hay imágenes en este split.")
        st.stop()

    # Navegación
    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)
    with col_nav1:
        if st.button("⏮️ Inicio"):
            st.session_state.idx = 0
    with col_nav2:
        if st.button("◀️ Anterior"):
            st.session_state.idx = max(0, st.session_state.idx - 1)
    with col_nav3:
        if st.button("Siguiente ▶️"):
            st.session_state.idx = min(len(sub) - 1, st.session_state.idx + 1)
    with col_nav4:
        if st.button("Fin ⏭️"):
            st.session_state.idx = len(sub) - 1

    idx = st.session_state.idx
    row = sub.iloc[idx]

    # Mostrar imagen y metadatos
    left, right = st.columns([2, 1])
    with left:
        st.image(row["path"], caption=f"{row['class']} | label={row['label']} | split={row['split']}", use_container_width=True)
    with right:
        st.markdown("### Etiqueta binaria")
        current = int(row["label"])  # 1 reciclable, 0 no reciclable
        st.write(f"Actual: {'Reciclable (1)' if current == 1 else 'No Reciclable (0)'}")
        if st.button("Marcar como Reciclable (1)"):
            df.loc[df["path"] == row["path"], "label"] = 1
            st.session_state.df = df
        if st.button("Marcar como No Reciclable (0)"):
            df.loc[df["path"] == row["path"], "label"] = 0
            st.session_state.df = df
        st.divider()
        if st.button("Guardar CSV"):
            df.to_csv(args.csv, index=False)
            st.success("CSV guardado")
        st.caption("Consejo: usa los botones de navegación para revisar rápidamente.")

    st.markdown(f"Ítem {idx+1} de {len(sub)} en split '{st.session_state.split}'")


if __name__ == "__main__":
    main()
