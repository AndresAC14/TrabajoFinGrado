import pandas as pd
from pandas import json_normalize
import json
pd.set_option('display.max_columns', 15)

df = pd.read_json('data.json')
print(df.head())



# Problema, solo saca las 21 filas y esto hay que ir modificando los datos antes de pasarlo a df