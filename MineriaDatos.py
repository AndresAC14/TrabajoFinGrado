# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
import json
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 300)

# Abre el fichero .json
with open('data.json') as json_file:
    data = json.load(json_file)


# Función para normalizar JSON
def normalize_json(item, level: int, parent_id=None, parent_name=None):

    result = [{
        'id': item.get('id'),
        'ClaseReal': item.get('name'),
        'Nivel': level,
        'sift': item.get('sift'),
        'index': item.get('index'),
        'parent_id': parent_id,
        'ClasePadre': parent_name
    }]

    if 'children' in item:
        level = level + 1
        for child in item['children']:
            result.extend(normalize_json(child, level=level, parent_id=item.get('id'), parent_name=item.get('name')))

    return result


# Aplanar el json
normalized_data = normalize_json(data, level=0)

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


# Transformaciones basicas
df_flat.drop(df_flat[df_flat['id'] == 'fall11'].index, inplace=True)
# df_flat.loc[df_flat['parent_id'] == 'fall11', 'parent_id'] = 'root_id'
df_flat.loc[df_flat['ClasePadre'] == 'ImageNet 2011 Fall Release', 'ClasePadre'] = 'root_name'
#df_flat = df_flat.drop(['id', 'sift', 'index', 'parent_id'], axis=1)
df_flat = df_flat.drop(['sift', 'index'], axis=1)
df_flat['ClaseReal'] = df_flat['ClaseReal'].apply(lambda cadena: cadena.split(',')[0])
df_flat['ClaseReal'] = df_flat['ClaseReal'].str.lower()
df_flat['ClasePadre'] = df_flat['ClasePadre'].apply(lambda cadena: cadena.split(',')[0])
df_flat['ClasePadre'] = df_flat['ClasePadre'].str.lower()

# print(df_flat.head(30))

# HAY QUE QUITAR DUPLICADOS!!!! -> antes de nada porque sino al buscar los padres o abuelos se puede volver loco y aun asi se puede liar bastante
df_flat = df_flat.drop_duplicates(subset=['ClaseReal'],keep='first')


print("################################################################################################################")
# Funcion para establecer los abuelos

# Inicializamos la columna 'ClaseAbuelo'
df_flat['ClaseAbuelo'] = "no_abuelo"
df_flat['grandparent_id'] = 'no_id'

# Iterar sobre cada fila del DataFrame
for idx, row in df_flat.iterrows():
    # Obtenemos el id del padre actual
    parent_id = row['parent_id']
    #print(f'El parent id \n {parent_id}')

    if pd.notna(parent_id):  # Verificamos que no sea NaN
        # Buscar el ID del abuelo
        grandparent_id = df_flat.loc[df_flat['id'] == parent_id, 'parent_id'].values
        grandparent_name = df_flat.loc[df_flat['id'] == parent_id, 'ClasePadre'].values
        #print(f'El grandparent id \n {grandparent_id} y {grandparent_name}')

        if grandparent_name.size > 0:
            df_flat.at[idx, 'ClaseAbuelo'] = grandparent_name
            df_flat.at[idx, 'grandparent_id'] = grandparent_id

print("################################################################################################################")

nuevo_orden = ['Nivel', 'id', 'ClaseReal', 'parent_id', 'ClasePadre', 'grandparent_id', 'ClaseAbuelo']
df_flat = df_flat[nuevo_orden]

# df_flat.to_csv('Jerarquia_Clases.csv', index=False)

# print(df_flat.head(10))
# print(df_flat.tail(10))

# Quiero duplicar el df pero ahora en lugar de ClaseReal sea ClasePredPrime (para hacer el merge) ClasePadre -> ClasePredPadre, ClaseAbuelo -> ClasePredAbuelo
# Diccionario para renombrar las columnas
nuevos_nombres = {
    'ClaseReal': 'ClasePredPrime',
    'ClasePadre': 'ClasePredPadre',
    'ClaseAbuelo': 'ClasePredAbuelo'
}

# Renombrar las columnas del DataFrame
df_renombrado = df_flat.rename(columns=nuevos_nombres)
df_renombrado = df_renombrado.drop(['Nivel', 'id', 'parent_id', 'grandparent_id'], axis=1)
# df_renombrado.to_csv('Jerarquia_Clases_Pred.csv', index=False)

# print(df_renombrado.head(15))

print("################################################################################################################")

print(df_flat.sample(10))