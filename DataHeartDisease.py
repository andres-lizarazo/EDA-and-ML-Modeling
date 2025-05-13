#%%
import pandas as pd

# Cambia la ruta si tu archivo está en otra carpeta
ruta_archivo = 'heart_disease_dataset.csv'

# Leer el archivo CSV
df = pd.read_csv(ruta_archivo)

# Mostrar las primeras filas del DataFrame
df.head()
# %%
