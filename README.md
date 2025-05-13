<h1 align="center">🧠 Heart Disease Prediction - EDA & ML Modeling</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data_Analysis-green?style=for-the-badge&logo=pandas" />
  <img src="https://img.shields.io/badge/Scikit--learn-Modeling-orange?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Status-In_Progress-yellow?style=for-the-badge" />
</p>

---

## 🩺 Objetivo del Proyecto

Este proyecto tiene como propósito realizar un **Análisis Exploratorio de Datos (EDA)** y construir modelos de **Machine Learning** para predecir la presencia de enfermedad cardíaca en pacientes, usando un conjunto de datos clínicos simulados.

---

## 📁 Estructura del Proyecto

EDA-and-ML-Modeling/
│
├── heart_disease_dataset.csv # Dataset con variables clínicas
├── eda.py # Análisis exploratorio (script)
├── ml_modeling.py # Modelado de Machine Learning (script)
├── img/ # Visualizaciones exportadas



---

## 🔍 Descripción del Dataset

El conjunto de datos contiene variables relevantes como edad, género, colesterol, presión arterial, historial familiar, y estilo de vida. La variable objetivo es `Heart Disease` (1: Sí, 0: No).

| Variable               | Descripción                                          |
|------------------------|------------------------------------------------------|
| Age                   | Edad del paciente (años)                             |
| Gender                | Género (Male/Female)                                 |
| Cholesterol           | Nivel de colesterol (mg/dL)                          |
| Blood Pressure        | Presión arterial sistólica (mmHg)                    |
| Smoking               | Tabaquismo (Never/Former/Current)                    |
| Alcohol Intake        | Ingesta de alcohol (None/Moderate/Heavy)             |
| Exercise Hours        | Horas de ejercicio por semana                        |
| Family History        | Historial familiar de enfermedad cardíaca (Yes/No)   |
| Diabetes              | Estado diabético (Yes/No)                            |
| Obesity               | Estado de obesidad (Yes/No)                          |
| Stress Level          | Nivel de estrés (1 a 10)                             |
| Blood Sugar           | Azúcar en sangre en ayunas (mg/dL)                   |
| Exercise Induced Angina | Angina inducida por ejercicio (Yes/No)             |
| Chest Pain Type       | Tipo de dolor torácico                               |
| **Heart Disease**     | 0: No, 1: Sí (variable objetivo)                     |

---

## 📊 Análisis Exploratorio (EDA)

- Distribución de variables
- Matriz de correlación
- Análisis por subgrupos (género, edad, etc.)
- Visualizaciones guardadas en `/img`

> Ejecuta `eda.py` para generar gráficos automáticamente.

---

## 🤖 Modelado de Machine Learning

- Preparación de datos
- Codificación de variables categóricas
- Entrenamiento y evaluación de modelos:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Métricas utilizadas: Accuracy, ROC-AUC, Confusion Matrix

> Ejecuta `ml_modeling.py` para entrenar modelos y ver resultados.

---

## 🛠 Tecnologías Usadas

- `Python` · `Pandas` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `XGBoost`

---

## 📌 Estado del Proyecto

🚧 En desarrollo: se planea incluir notebooks interactivos y más visualizaciones.

---

## 📎 Autor

**Andrés Lizarazo**  usign Data from Kaggle Dataset 
