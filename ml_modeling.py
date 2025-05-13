#%%
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Leer el archivo CSV
df = pd.read_csv('heart_disease_dataset.csv')

# Variables categóricas
cat_cols = [
    'Gender', 'Smoking', 'Alcohol Intake', 'Family History',
    'Diabetes', 'Obesity', 'Exercise Induced Angina', 'Chest Pain Type'
]

# One-hot encoding para variables con más de dos categorías
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Separar variables predictoras y objetivo
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear carpeta para guardar imágenes
if not os.path.exists('img'):
    os.makedirs('img')

# 1. Regresión Logística
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)

# 2. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluación de modelos
def evaluar_modelo(y_true, y_pred, nombre):
    print(f"\nResultados para {nombre}:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_true, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_true, y_pred))
    # Guardar matriz de confusión
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de confusión - {nombre}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.savefig(f'img/matriz_confusion_{nombre}.png')
    plt.close()

evaluar_modelo(y_test, y_pred_logreg, 'LogisticRegression')
evaluar_modelo(y_test, y_pred_rf, 'RandomForest')

def plot_roc(y_true, y_pred_proba, nombre):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - {nombre}')
    plt.legend()
    plt.savefig(f'img/roc_{nombre}.png')
    plt.close()

# Para regresión logística
plot_roc(y_test, logreg.predict_proba(X_test_scaled)[:,1], 'LogisticRegression')
# Para Random Forest
plot_roc(y_test, rf.predict_proba(X_test)[:,1], 'RandomForest')
#%%
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title('Importancia de variables - Random Forest')
plt.savefig('img/importancia_variables_rf.png')
plt.close() 
# %%

scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5)
print(f'Accuracy promedio (Logistic Regression, CV=5): {scores.mean():.2f}')

param_grid = {'n_estimators': [50, 100, 200]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid.fit(X_train, y_train)
print('Mejores parámetros:', grid.best_params_)

# %%
