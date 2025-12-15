## ♻️ TrashNet Binary Classifier
### Evaluación III-IV Machine Learning TIEL26-101-8V-A

**✅ PROYECTO COMPLETADO Y EJECUTADO EXITOSAMENTE (Deep Learning + Producción)**

---

## 📘 Información Académica
- Estudiante: Christian Santibáñez Martínez  
- Profesor: Felipe Oyarzún  
- Institución: INACAP  
- Fecha: 15 de Diciembre, 2025  

---

## 📖 Descripción del Proyecto
Proyecto de **Deep Learning aplicado a visión por computador**, cuyo objetivo es clasificar imágenes de residuos en dos clases: **Reciclable (1)** y **No Reciclable (0)** usando el dataset público **TrashNet**. 

El trabajo integra todo el ciclo de vida de un modelo de Deep Learning:
- Construcción de un clasificador binario basado en **CNN pre-entrenadas** (ResNet-18 / MobileNetV3-Small) con **transfer learning y fine-tuning**.
- Generación y aumento del conjunto de datos (re-etiquetado binario, splits estratificados y data augmentation geométrico/fotométrico).
- Entrenamiento, validación cruzada y evaluación final en un conjunto de test independiente.
- **Exportación del modelo a ONNX** y desarrollo de un **servicio de inferencia con FastAPI**, pensado para producción en entornos locales o cloud.
- Configuración de **integración continua (CI)** con GitHub Actions para validar automáticamente la construcción del modelo y la exportación a ONNX.

---

## 🎯 Objetivos de Aprendizaje
- Diseñar e implementar un modelo de **Deep Learning** basado en redes convolucionales y transferencia de aprendizaje para clasificación de imágenes.  
- Configurar y comparar estrategias de **fine-tuning** (entrenamiento completo del backbone vs. entrenamiento solo de la cabeza).  
- Generar y aumentar un conjunto de datos para Deep Learning, aplicando **splits estratificados y data augmentation** para mejorar la generalización.  
- Evaluar rigurosamente el desempeño del modelo (Accuracy, Precision, Recall, F1, ROC-AUC) mediante validación cruzada y test independiente.  
- Implementar el modelo en **modo de producción** usando exportación a ONNX y un servicio de inferencia con FastAPI, considerando eficiencia y uso de recursos.  
- Incorporar una **integración continua básica (CI)** que ejecute tests sobre el modelo y el proceso de exportación, asegurando reproducibilidad y mantenibilidad del proyecto.

---

## 🏗️ Modelo y Configuración (Deep Learning)
```
Entrada (224x224 RGB) → CNN pre-entrenada (ResNet-18 / MobileNetV3-Small)
Pérdida: BCEWithLogitsLoss (con class weights)
Optimizador: AdamW (lr=3e-4, wd=1e-4) + Cosine LR / ReduceLROnPlateau
Early Stopping: paciencia=5

Estrategia de fine-tuning (config.yaml → training.finetune_strategy):
- "full": entrena todo el backbone (fine-tuning completo)
- "head": congela el backbone y entrena solo la última capa (head-only)
```

---

## 📊 Dataset
- Origen: TrashNet (MIT).  
- Clases originales (6): glass, paper, cardboard, plastic, metal, trash.  
- Mapeo binario: reciclable={glass,paper,cardboard,plastic,metal} → 1; no reciclable={trash} → 0.  
- Estructura esperada: `data/raw/dataset-resized/<clase>/*.jpg`.  
- CSV con splits: `data/interim/labels.csv` (2527 filas; train/val/test ≈ 70/15/15).

---

## ⚙️ Cómo Ejecutar el Proyecto
1) Crear entorno e instalar dependencias (Windows/PowerShell):
```powershell
python -m venv .venv
\.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
2) Preparar dataset y generar CSV:
```powershell
python -m src.data.prepare_dataset --raw_dir data/raw/dataset-resized --out_csv data/interim/labels.csv --test_size 0.15 --val_size 0.15 --seed 42
```
3) Herramienta de etiquetado (opcional):
```powershell
streamlit run tools/label_tool_streamlit.py -- --csv data/interim/labels.csv
```
4) Entrenamiento (Deep Learning):
```powershell
python -m src.train.train --config config.yaml
```
5) Validación cruzada (5-fold):
```powershell
python -m src.train.cross_validate --config config.yaml
```
6) Figuras para el informe (curvas + matriz + ROC):
```powershell
python -m src.train.evaluate --exp_dir experiments/exp_YYYYMMDD_HHMMSS --config config.yaml --out_dir report/figuras
```

7) Exportar el mejor modelo a ONNX (para producción):
```powershell
python -m src.models.export_onnx --exp_dir experiments/exp_YYYYMMDD_HHMMSS --config config.yaml
```
Genera `experiments/exp_YYYYMMDD_HHMMSS/model.onnx` usando la configuración guardada.

8) Servicio de inferencia (FastAPI + ONNX, local):
```powershell
uvicorn serving.api_fastapi:app --reload
```
- Ir a `http://127.0.0.1:8000/` para ver el estado del servicio.
- Ir a `http://127.0.0.1:8000/docs` para abrir la UI automática (Swagger) y probar el endpoint `POST /predict` subiendo imágenes.

---

## 📈 Resultados del Proyecto
### Validación Cruzada (5 folds)
- Accuracy: 0.9423 ± 0.0115  
- Precision macro: 0.7435 ± 0.0255  
- Recall macro: 0.9490 ± 0.0153  
- F1 macro: 0.8068 ± 0.0233  
- ROC-AUC: 0.9818 ± 0.0082  

### Test Final (modelo entrenado)
- Accuracy: 0.9658  
- F1 macro: 0.8398  
- ROC-AUC: 0.9674  

---

## 📂 Estructura del Proyecto
```
Evaluación II Machine Learning TIEL26-101-8V-A/
├── config.yaml                       # Configuración del pipeline
├── requirements.txt                  # Dependencias
├── README.md                         # Este documento
├── .gitignore                        # Configuración de Git
├── predicciones_trash.csv            # CSV de inferencia por carpeta [GENERADO]
│
├── data/
│   ├── raw/                          # Colocar TrashNet (IGNORADO)
│   ├── interim/                      # CSV con splits
│   └── processed/                    # Procesados (IGNORADO)
│
├── src/
│   ├── data/                         # prepare_dataset, dataset, augmentations
│   ├── models/                       # build_model, export_onnx
│   ├── train/                        # train, cross_validate, evaluate
│   └── utils/                        # seed, metrics, config
│
├── tools/
│   └── label_tool_streamlit.py       # Herramienta de etiquetado
│
├── serving/
│   └── api_fastapi.py                # Servicio de inferencia (FastAPI + ONNX)
│
├── tests/
│   └── test_model_and_export.py      # Tests: forma del modelo + exportación ONNX
│
├── .github/
│   └── workflows/
│       └── ci.yml                    # CI (GitHub Actions) ejecuta pytest en cada push/PR
│
├── experiments/                      # Artefactos (pesos ignorados; JSON visibles)
│
└── report/
    └── figuras/                      # Imágenes para el informe
```

---

## 🔎 Análisis de Errores (resumen)
- Recall macro alto sugiere buena sensibilidad en ambas clases.  
- Precision macro menor indica algunos falsos positivos (umbral ajustable).  
- Oportunidad: threshold tuning o focal loss según requerimientos.

---

## 📊 Visualizaciones Incluidas
1. `report/figuras/loss_curves.png` — Curvas de pérdida train/val.  
2. `report/figuras/val_metrics_curves.png` — Accuracy/F1 val por época.  
3. `report/figuras/confusion_matrix_test.png` — Matriz de confusión test.  
4. `report/figuras/roc_curve_test.png` — Curva ROC test.  

---

## 🧾 Reproducibilidad
- Configuración centralizada en `config.yaml`.  
- Semillas fijadas (`src/utils/seed.py`).  
- `experiments/`: JSON visibles; pesos `.pt/.pth` ignorados.  
- Exportación a ONNX reproducible vía `src/models/export_onnx.py`.  
- CI mínima en GitHub Actions (`.github/workflows/ci.yml`) que instala dependencias y ejecuta `pytest`.

---

## 🚀 Demo rápida (inferencia)
Usa el mejor modelo entrenado para predecir si una imagen es reciclable.

1) Imagen única:
```powershell
python -m src.train.predict --exp_dir experiments/exp_YYYYMMDD_HHMMSS --image "data/raw/dataset-resized/plastic/plastic390.jpg"
```

2) Carpeta completa (genera CSV opcional):
```powershell
python -m src.train.predict --exp_dir experiments/exp_YYYYMMDD_HHMMSS --dir "data/raw/dataset-resized/trash" --csv_out "predicciones_trash.csv"
```

Opcionales:
- `--threshold 0.5` para ajustar el umbral de clasificación.
- `--image_size 224` para cambiar el tamaño de entrada.

---

## 📚 Citación y Licencias
- TrashNet — MIT License (https://huggingface.co/datasets/garythung/trashnet).  
- Este proyecto con fines académicos.

