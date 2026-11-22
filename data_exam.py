import pandas as pd

df = pd.read_csv("data/raw/Sample - Superstore.csv", encoding="latin-1")

print(df.shape)     # (filas, columnas)
print(df.head())    # primeras 5 filas
df.info()           # columnas, tipos y valores no nulos