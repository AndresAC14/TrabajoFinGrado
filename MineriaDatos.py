# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
import json
pd.set_option('display.max_columns', 15)

# Abre el fichero .json
with open('data.json') as json_file:
    data = json.load(json_file)


# Función para aplanar la estructura
def flatten_children(item, parent_id=None, parent_name=None):
    result = []
    if 'children' in item:
        for child in item['children']:
            result.extend(flatten_children(child, parent_id=item.get('id'), parent_name=item.get('name')))

    result.append({
        'id': item.get('id'),
        'name': item.get('name'),
        'sift': item.get('sift'),
        'index': item.get('index'),
        'parent_id': parent_id,
        'parent_name': parent_name
    })

    return result


# Aplanar el json
flattened_data = flatten_children(data)

# Crea un DataFrame a partir de la lista de diccionarios
df_flat = pd.DataFrame(flattened_data)
print(df_flat)

# Informacion del dataframe
# Filas y columnas (2155 filas y 6 columnas)
print(df_flat.shape)

# Columnas -> sift index se pueden quitar
columnas = df_flat.columns
print(columnas)

# Informacion/tipo de las columnas del df
print(df_flat.info())

'''
Data columns (total 6 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   id           2155 non-null   object 
 1   name         2155 non-null   object 
 2   sift         1258 non-null   object ---> NO se que es esta columna (aunque creo puede indicar si es hoja o nodo)
 3   index        1258 non-null   float64 ---> NO se que es esta columna
 4   parent_id    2154 non-null   object 
 5   parent_name  2154 non-null   object 
dtypes: float64(1), object(5)
'''

# Primeras filas
# print(df_flat.head(5))

# print(df_flat[df_flat['id'] == 'n03770679'])

# Contar la frecuencia de cada valor en la columna 'id'
frecuencia_id = df_flat['id'].value_counts()
print(frecuencia_id.head(10))

# Filtrar las filas donde el valor de 'id' aparezca más de x veces
filas_repetidas = df_flat[df_flat['id'].isin(frecuencia_id[frecuencia_id > 3].index)]
print(filas_repetidas)

# 743 cosas hay repetidas -> misma cosa, distinto padre, aunque la que mas se repite son 4 veces




