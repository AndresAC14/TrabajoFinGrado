# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches

# Opciones de configuración
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 400)

# Configuración global del tamaño de la fuente
plt.rcParams.update({'font.size': 18})


# Importar el dataframe
df = pd.read_csv('Conjunto_Entrenamiento_10000.csv')

print(df.head(10))

# Diccionario para cambiar los nombres de las columnas
nuevos_nombres = {
    'Unnamed: 0': 'Index',
    'Media': 'Mean',
    'Mediana': 'Median',
    'DesvTipica': 'StdDev',
    'Algoritmo': 'Algorithm',
    'MediaPrime': 'MeanPrime',
    'MedianaPrime': 'MedianPrime',
    'DesvTipicaPrime': 'StdDevPrime',
    'ClaseReal': 'RealClass',
    'ClasePred': 'PredClass',
    'ProbClasePred': 'PredClassProb',
    'ClasePredPrime': 'PredPrimeClass',
    'ProbClasePredPrime': 'PredPrimeClassProb',
    'r': 'r'
}

# Cambiar los nombres de las columnas
df.rename(columns=nuevos_nombres, inplace=True)

algoritmo_map = {
    'Ecualización del histograma': 'Histogram Equalization',
    'Ecualización del histograma adaptativa limitada por contraste': 'Adaptive Histogram Equalization Limited by Contrast',
    'Corrección de gamma': 'Gamma Correction',
    'Transformación logarítmica': 'Log Transformation'
}

df['Algorithm'] = df['Algorithm'].map(algoritmo_map)

# Mostrar el DataFrame con los nuevos nombres de columnas
print(df)

#df.to_csv('Training_Set.csv', index=False)