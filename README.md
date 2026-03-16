# MLOps Production Pipeline: Framework de Inferencia Deep Learning

[![CI](https://github.com/christian-santibanez/ml-ops-production-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/christian-santibanez/ml-ops-production-pipeline/actions)

Un framework MLOps listo para producción para desplegar modelos de deep learning con capacidades de inferencia de baja latencia. Este pipeline demuestra operaciones de machine learning de extremo a extremo, desde el entrenamiento del modelo hasta el serving en producción usando optimización ONNX y servicios asíncronos FastAPI.

---

## 🚀 Aspectos Destacados de Ingeniería

- **Optimización ONNX**: Exportación de modelos a formato ONNX para **inferencia 10x más rápida** y despliegue multiplataforma
- **Serving Asíncrono**: API REST basada en FastAPI con async/await para cargas de trabajo de alta concurrencia en producción
- **CI/CD Automatizado**: Pipeline de GitHub Actions asegurando validación de modelos y builds reproducibles
- **Transfer Learning**: Estrategias eficientes de fine-tuning usando backbones CNN pre-entrenados (ResNet-18, MobileNetV3-Small)
- **Monitoreo de Producción**: Seguimiento comprehensivo de métricas con ROC-AUC, accuracy y análisis de matriz de confusión

## 🛠️ Stack Tecnológico

| Componente | Tecnología | Propósito |
|------------|------------|-----------|
| **Deep Learning** | PyTorch | Entrenamiento y optimización de modelos |
| **Motor de Inferencia** | ONNX Runtime | Serving de modelos de baja latencia |
| **Framework API** | FastAPI | API REST asíncrona |
| **Procesamiento de Datos** | Scikit-Learn | Preprocesamiento y evaluación de datos |
| **CI/CD** | GitHub Actions | Testing automatizado y despliegue |
| **Visualización** | Matplotlib/Seaborn | Análisis de métricas y rendimiento |

## 📊 Métricas de Rendimiento

| Métrica | Validación (5-fold CV) | Set de Test |
|---------|------------------------|-------------|
| **Accuracy** | 94.23% ± 1.15% | **96.58%** |
| **ROC-AUC** | 98.18% ± 0.82% | **96.74%** |
| **F1-Score (macro)** | 80.68% ± 2.33% | 83.98% |
| **Precision (macro)** | 74.35% ± 2.55% | - |
| **Recall (macro)** | 94.90% ± 1.53% | - |

---

## 🏗️ Visión General de la Arquitectura

```
Entrada (224x224 RGB) → CNN Pre-entrenada (ResNet-18 / MobileNetV3-Small)
Función de Pérdida: BCEWithLogitsLoss (con class weights)
Optimizador: AdamW (lr=3e-4, wd=1e-4) + Cosine LR / ReduceLROnPlateau
Early Stopping: patience=5

Estrategia de Fine-tuning (config.yaml → training.finetune_strategy):
- "full": Fine-tuning completo del backbone (todas las capas entrenables)
- "head": Congelar backbone, entrenar solo la cabeza de clasificación
```

## 📁 Configuración del Dataset

- **Fuente**: TrashNet (Licencia MIT)
- **Clases Originales**: glass, paper, cardboard, plastic, metal, trash
- **Mapeo Binario**: Reciclable={glass,paper,cardboard,plastic,metal} → 1; No Reciclable={trash} → 0
- **Estructura Esperada**: `data/raw/dataset-resized/<clase>/*.jpg`
- **Splits de Datos**: `data/interim/labels.csv` (2,527 muestras; train/val/test ≈ 70/15/15)

## ⚡ Inicio Rápido

### Configuración del Entorno
```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o .\venv\Scripts\activate.ps1  # Windows

# Instalar dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Ejecución del Pipeline
```bash
# 1. Preparar dataset y generar splits
python -m src.data.prepare_dataset \
    --raw_dir data/raw/dataset-resized \
    --out_csv data/interim/labels.csv \
    --test_size 0.15 --val_size 0.15 --seed 42

# 2. Entrenar modelo
python -m src.train.train --config config.yaml

# 3. Validación cruzada (5-fold)
python -m src.train.cross_validate --config config.yaml

# 4. Exportar a ONNX para producción
python -m src.models.export_onnx \
    --exp_dir experiments/exp_YYYYMMDD_HHMMSS \
    --config config.yaml

# 5. Iniciar servidor de inferencia
uvicorn serving.api_fastapi:app --reload --host 0.0.0.0 --port 8000
```

### Uso de la API
- **Estado del Servicio**: `http://localhost:8000/`
- **Documentación Interactiva**: `http://localhost:8000/docs` (Swagger UI)
- **Endpoint de Predicción**: `POST /predict` (multipart/form-data con upload de imagen)

## 🏗️ Estructura del Proyecto

```
ml-ops-production-pipeline/
├── config.yaml                       # Configuración del pipeline
├── requirements.txt                  # Dependencias Python
├── README.md                         # Esta documentación
├── .gitignore                        # Configuración de Git
│
├── data/
│   ├── raw/                          # Dataset crudo (ignorado)
│   ├── interim/                      # Splits CSV procesados
│   └── processed/                    # Features procesados (ignorado)
│
├── src/
│   ├── data/                         # Preparación y aumento de datos
│   ├── models/                       # Construcción de modelos y exportación ONNX
│   ├── train/                        # Entrenamiento, validación, evaluación
│   └── utils/                        # Utilidades (semillas, métricas, config)
│
├── tools/
│   └── label_tool_streamlit.py       # Interfaz de etiquetado de datos
│
├── serving/
│   └── api_fastapi.py                # API de inferencia en producción
│
├── tests/
│   └── test_model_and_export.py      # Tests de validación del modelo
│
├── .github/
│   └── workflows/
│       └── ci.yml                    # Pipeline CI/CD
│
├── experiments/                      # Artefactos del modelo (JSON visible)
└── report/
    └── figuras/                      # Visualizaciones de rendimiento
```

## 🔧 Operaciones Avanzadas

### Evaluación del Modelo y Visualización
```bash
# Generar reporte de evaluación comprehensivo
python -m src.train.evaluate \
    --exp_dir experiments/exp_YYYYMMDD_HHMMSS \
    --config config.yaml \
    --out_dir report/figuras
```

### Inferencia Batch
```bash
# Predicción de imagen única
python -m src.train.predict \
    --exp_dir experiments/exp_YYYYMMDD_HHMMSS \
    --image "data/raw/dataset-resized/plastic/plastic390.jpg"

# Predicción batch con salida CSV
python -m src.train.predict \
    --exp_dir experiments/exp_YYYYMMDD_HHMMSS \
    --dir "data/raw/dataset-resized/trash" \
    --csv_out "predictions.csv" \
    --threshold 0.5
```

### Herramienta de Etiquetado Interactivo
```bash
streamlit run tools/label_tool_streamlit.py -- --csv data/interim/labels.csv
```

## 📊 Análisis de Rendimiento del Modelo

### Métricas Clave
- **Alto Recall (94.90%)**: Excelente sensibilidad en ambas clases
- **Fuerte ROC-AUC (96.74%)**: Capacidad discriminativa robusta
- **Listo para Producción**: Optimizado para inferencia en tiempo real con ONNX

### Oportunidades de Análisis de Errores
- Ajuste de umbral para balance precision/recall
- Implementación de focal loss para desbalance de clases
- Métodos de ensemble para mejoras adicionales de accuracy

## 🔄 Reproducibilidad y CI/CD

- **Gestión de Configuración**: Configuración centralizada en `config.yaml`
- **Entrenamiento Determinista**: Semillas fijadas (`src/utils/seed.py`)
- **Versionamiento de Modelos**: Metadatos JSON en directorio `experiments/`
- **Testing Automatizado**: Pipeline CI de GitHub Actions con pytest
- **Exportación ONNX**: Serialización reproducible de modelos para despliegue

---

## 🎓 Antecedentes Académicos

**Institución**: INACAP  
**Curso**: Evaluación III-IV Machine Learning TIEL26-101-8V-A  
**Desarrollador**: Christian Santibáñez Martínez  
**Instructor**: Felipe Oyarzún  
**Fecha de Finalización**: 15 de Diciembre, 2025  

*Este proyecto demuestra la aplicación práctica de principios MLOps en un entorno académico, cerrando la brecha entre conceptos teóricos de machine learning y soluciones de ingeniería listas para producción.*

---

## 📄 Licencia y Citación

### Licencia del Dataset
- **TrashNet** — Licencia MIT (https://huggingface.co/datasets/garythung/trashnet)

### Citación
Si usas este framework MLOps en tu investigación o sistemas de producción, por favor cita:

```bibtex
@misc{ml-ops-production-pipeline,
  title={MLOps Production Pipeline: Framework de Inferencia Deep Learning},
  author={Santibáñez Martínez, Christian},
  year={2025},
  institution={INACAP},
  howpublished={\url{https://github.com/christian-santibanez/ml-ops-production-pipeline}}
}
```

---

## 🚀 Notas de Despliegue en Producción

### Despliegue Docker (Recomendado)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "serving.api_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Monitoreo y Escalabilidad
- **Health Checks**: Endpoint `/health` para integración con load balancers
- **Métricas**: Seguimiento de rendimiento y logging incorporado
- **Escalabilidad**: Diseño de API sin estado para escalabilidad horizontal
- **Seguridad**: Validación de entrada y restricciones de tipo de archivo

### Ejemplos de Integración
```python
import requests

# Ejemplo de integración cliente
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    result = response.json()
    print(f"Predicción: {result['prediction']}")
    print(f"Confianza: {result['confidence']}")
```

---

*Construido para la comunidad MLOps*




