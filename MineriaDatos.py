import pandas as pd
import json
pd.set_option('display.max_columns', 15)

# Reemplaza 'tu_archivo.json' con la ruta de tu archivo JSON
with open('data.json') as json_file:
    data = json.load(json_file)


# Función para aplanar la estructura 'children'
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
print(df_flat)
