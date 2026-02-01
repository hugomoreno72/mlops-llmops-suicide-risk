# Prevención de Riesgo Suicida - MLOps & LLMOps Project

Este proyecto despliega una solución para la detección de comentarios sospechosos de tendencia suicida, combinando modelos clásicos de Machine Learning (Scikit-Learn + MLflow) con modelos de Hugging Face.

## 🚀 Estructura del Proyecto

* `mlops_llmops_project.ipynb`: Notebook con el análisis exploratorio (EDA), preprocesamiento y entrenamiento.
* `funciones.py`: Script de entrenamiento parametrizado para registro de experimentos en MLflow.
* `main.py`: FastAPI que integra 5 endpoints funcionales.

## 🛠️ Tecnologías Utilizadas

* FastAPI & Uvicorn.
* MLflow para el seguimiento de métricas y versionado de modelos.
* Scikit-Learn (Logistic Regression + TfidfVectorizer).
* Hugging Face Transformers.
