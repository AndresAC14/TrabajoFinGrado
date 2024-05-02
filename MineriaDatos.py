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


# Función para normalizar JSON
def normalize_json(item, parent_id=None, parent_name=None):
    result = []
    if 'children' in item:
        for child in item['children']:
            result.extend(normalize_json(child, parent_id=item.get('id'), parent_name=item.get('name')))

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
normalized_data = normalize_json(data)

# Crea un DataFrame a partir de la lista de diccionarios
df_flat = pd.DataFrame(normalized_data)

# Informacion del dataframe
# Filas y columnas (2154 filas y 6 columnas)
# print(df_flat.shape)

# Columnas -> sift index se pueden quitar
# columnas = df_flat.columns
# print(columnas)

# Informacion/tipo de las columnas del df
# print(df_flat.info())

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

df_flat.drop(df_flat[df_flat['id'] == 'fall11'].index, inplace=True)
df_flat.loc[df_flat['parent_id'] == 'fall11', 'parent_id'] = 'root_id'
df_flat.loc[df_flat['parent_name'] == 'ImageNet 2011 Fall Release', 'parent_name'] = 'root_name'
df_flat = df_flat.drop(['sift', 'index'], axis=1)
print(df_flat.head(10))

'''
# Contar la frecuencia de cada valor en la columna 'id'
frecuencia_id = df_flat['id'].value_counts()
print('IDs repetidos')
print(frecuencia_id.head(10))

# Filtrar las filas donde el valor de 'id' aparezca más de x veces
filas_repetidas = df_flat[df_flat['id'].isin(frecuencia_id[frecuencia_id > 3].index)]
print(filas_repetidas)

# 743 cosas hay repetidas -> misma cosa, distinto padre, aunque la que mas se repite son 4 veces
'''

# Interesa sacar los niveles de cada fila
# Al menos los raiz y los nodos hoja, es decir, nadie tiene como parent_id ese id

# Lista nodos raiz -> OK
# nodos_raiz = df_flat[df_flat['parent_id'] == 'root_id']
# nodos_raiz = nodos_raiz.drop(['parent_id', 'parent_name'], axis=1)
# print(nodos_raiz.head(5))


# Lista nodos hoja -> version sucia doble for
for i in range(len(df_flat)):
    for j in range(len(df_flat)):
        # if df_flat['id'] nose