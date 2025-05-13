#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Leer el archivo CSV
ruta_archivo = 'heart_disease_dataset.csv'
df = pd.read_csv(ruta_archivo)

# Mostrar las primeras filas del DataFrame
print("Primeras filas del DataFrame:")
print(df.head())

# Información general del DataFrame
print("\nInformación general del DataFrame:")
print(df.info())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df.describe())

# Verificar valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Visualizar la distribución de la variable objetivo (si existe)
if 'target' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df)
    plt.title('Distribución de la variable objetivo')
    plt.savefig('img/distribucion_target.png')
    plt.close()

# Convertir columnas no numéricas a numéricas si es posible
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.Categorical(df[col]).codes

# Visualizar la correlación entre variables numéricas
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de correlación')
plt.savefig('img/matriz_correlacion.png')
plt.show()
plt.close()

#%%
# Visualizar la distribución de algunas variables numéricas
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols[:4]:  # Mostrar solo las primeras 4 columnas numéricas
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribución de {col}')
    plt.savefig(f'img/distribucion_{col}.png')
    plt.show()
    plt.close() 