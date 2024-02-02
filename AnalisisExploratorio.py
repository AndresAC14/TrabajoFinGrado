# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.set_option('display.max_columns', 15)

# Importar el dataframe
df = pd.read_csv('Conjunto_Entrenamiento_10000.csv')

# Eliminar primera columna porque se pone al importar el csv pero no nos sirve
df.drop(df.columns[0], axis=1, inplace=True)

# Filas y columnas (40000 filas y 13 columnas)
# print(df.shape)

# Informacion/tipo de las columnas del df
#print(df.info())

# Primeras filas
# print(df.head(5))

# Columnas
# columnas = df.columns
# print(columnas)

# Algoritmos
# algoritmos = df['Algoritmo'].unique()
# print("Algoritmos que manejamos: ", algoritmos)

'''
#ClaseReal mas comun grafico
print(df["ClaseReal"].value_counts())
df["ClaseReal"].value_counts()[:10].plot(kind='bar')
plt.title("Clase Real mas comun")
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.show()
'''