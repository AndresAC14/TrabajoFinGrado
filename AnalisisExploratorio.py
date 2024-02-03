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
# print(df.info())

# Primeras filas
print(df.head(5))

# Columnas
# columnas = df.columns
# print(columnas)

# Algoritmos
# algoritmos = df['Algoritmo'].unique()
# print("Algoritmos que manejamos: ", algoritmos)

# ClaseReal, ClasePred, ProbClasePred, ClasePredPrime, ProbClasePredPrime

'''
# ClaseReal mas comun grafico
print(df['ClaseReal'].value_counts())
df['ClaseReal'].value_counts()[:10].plot(kind='bar')
plt.title('Clase Real mas comun')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.show()
'''

'''
# ClasePred mas comun grafico
print(df['ClasePred'].value_counts())
df['ClasePred'].value_counts()[:10].plot(kind='bar')
plt.title('ClasePred mas comun')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.show()
'''

'''
# ClasePredPrime mas comun grafico
print(df['ClasePredPrime'].value_counts())
df['ClasePredPrime'].value_counts()[:10].plot(kind='bar')
plt.title('ClasePredPrime mas comun')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.show()
'''

########################################################################################################################
################################## LO QUE HAY AQUI DENTRO NO ESTA BIEN #################################################
########################################################################################################################

'''
# Enfrentar ClaseReal con ClasePred y/o ClasePredPrime (con heatmap)
# Sacar todos los valores unicos de todas las columnas de clase, ya que pueden repetirse o no y ser totalmente distintos
# Obtener los valores únicos de cada columna
valores_ClasePred = df['ClasePred'].unique()
valores_ClaseReal = df['ClaseReal'].unique()
valores_ClasePredPrime = df['ClasePredPrime'].unique()

# Combina todos los valores únicos sin repetición
valores_unicos_totales = set(valores_ClaseReal) | set(valores_ClasePred) | set(valores_ClasePredPrime)

# Puedes imprimir o trabajar con la variable valores_unicos_totales según tus necesidades
# print(valores_unicos_totales)

# Crear un DataFrame a partir del conjunto de valores únicos
df_valores_unicos = pd.DataFrame(list(valores_unicos_totales), columns=['Clases'])

# Puedes imprimir o trabajar con el nuevo DataFrame df_valores_unicos según tus necesidades
# print(df_valores_unicos)

# print(df_valores_unicos.value_counts())

# (df_valores_unicos.shape)
'''

'''
# Crear un gráfico de dispersión
plt.scatter(df['ClaseReal'], df['ClasePred'])

# Agregar etiquetas y título al gráfico
plt.xlabel('ClaseReal')
plt.ylabel('ClasePred')
plt.title('Relación entre ClaseReal y ClasePred')

# Mostrar el gráfico
plt.show()
'''


'''
# Crear una matriz de correlación entre las columnas ClaseReal y ClasePred
correlation_matrix = pd.crosstab(df['ClaseReal'], df['ClasePred'])

# Crear un heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='inferno', fmt='g')
plt.xlabel('ClasePred')
plt.ylabel('ClaseReal')
plt.title('Heatmap de Relación entre ClaseReal y ClasePred')
plt.show()
'''


'''
# Filtrar las filas donde ClaseReal y ClasePred no son iguales
df_filtered = df[df['ClaseReal'] == df['ClasePred']]

# Crear un gráfico de dispersión con los datos filtrados
plt.scatter(df_filtered['ClaseReal'], df_filtered['ClasePred'])

# Agregar etiquetas y título al gráfico
plt.xlabel('ClaseReal')
plt.ylabel('ClasePred')
plt.title('Relación entre ClaseReal y ClasePred (Filtrado)')

# Mostrar el gráfico
plt.show()
'''
########################################################################################################################
########################################################################################################################


'''
# Porcentaje de acierto, es decir, (Real == Pred) / Real

# Sacar todos los valores unicos de todas las columnas de clase, ya que pueden repetirse o no y ser totalmente distintos
# Obtener los valores únicos de cada columna
valores_ClasePred = df['ClasePred'].unique()
valores_ClaseReal = df['ClaseReal'].unique()
valores_ClasePredPrime = df['ClasePredPrime'].unique()

# Combina todos los valores únicos sin repetición
valores_unicos_totales = set(valores_ClaseReal) | set(valores_ClasePred) | set(valores_ClasePredPrime)

# Puedes imprimir o trabajar con la variable valores_unicos_totales según tus necesidades
# print(valores_unicos_totales)

# Crear un DataFrame a partir del conjunto de valores únicos
df_Porcentaje1 = pd.DataFrame(list(valores_unicos_totales), columns=['Clases'])


coincidencias_por_fila = (df['ClaseReal'].eq(df['ClasePred'])).sum(axis=1)

df_Porcentaje['Aciertos'] =
df_Porcentaje1['Total'] = df.groupby('ClaseReal')['ClasePred'].count()
df_Porcentaje1['Porcentaje'] = (df_Porcentaje1['Aciertos'] / df_Porcentaje1['Total']) * 100

df_Porcentaje1 = df_Porcentaje1.dropna()

print(df_Porcentaje1)
'''

print("################################################################################################################")
# print(df_filtered)

# Filtrar las filas donde ClaseReal y ClasePred no son iguales
df_filtered = df[df['ClaseReal'] == df['ClasePred']]

# Agrupar las que son iguales y contarlas
df_1 = df_filtered.groupby('ClaseReal')['ClasePred'].count().reset_index(name='Aciertos')
df_2 = df.groupby('ClaseReal')['ClasePred'].count().reset_index(name='Total')

df_Res = pd.merge(df_1, df_2, on='ClaseReal', how='left')

df_Res['Porcentaje'] = ((df_Res['Aciertos'] / df_Res['Total']) * 100).__round__(2)

print(df_Res)

# Ahora hacer una grafica que ponga la ClaseReal respecto al Porcentaje
sns.barplot(df_Res, x=df_Res['ClaseReal'], y=df_Res['Porcentaje'])
plt.show()

# Eso es demasiado grande, asi que mejor hacer la grafica de los 20 mas precisos por ejemplo o los 20 menos

