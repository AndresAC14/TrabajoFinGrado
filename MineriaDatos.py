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


# Obtén la lista de diccionarios aplanada
flattened_data = flatten_children(data)

# Crea un DataFrame a partir de la lista de diccionarios
df_flat = pd.DataFrame(flattened_data)

# Muestra el DataFrame resultante
# print(df_flat)

# Informacion del dataframe
# Filas y columnas (2155 filas y 6 columnas)
print(df_flat.shape)

# Informacion/tipo de las columnas del df
print(df_flat.info())

# Primeras filas
# print(df_flat.head(5))

# Columnas -> sift index se pueden quitar
columnas = df_flat.columns
print(columnas)

