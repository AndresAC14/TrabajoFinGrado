# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.set_option('display.max_columns', 15)

# Importar el dataframe
df = pd.read_csv('Conjunto_Entrenamiento_10000.csv')

# Eliminar primera columna y la r porque no nos sirven
df.drop(df.columns[0], axis=1, inplace=True)
df.drop(df.columns[12], axis=1, inplace=True)

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

# CLASE REAL RESPECTO A CLASE PREDICHA

# Filtrar las filas donde ClaseReal y ClasePred son iguales
df_filtered = df[df['ClaseReal'] == df['ClasePred']]
# print(df_filtered)

# Agrupar las que son iguales y contarlas
df_1 = df_filtered.groupby('ClaseReal')['ClasePred'].count().reset_index(name='Aciertos')
# print('df1',df_1.head(2))

df_2 = df.groupby('ClaseReal')['ClasePred'].count().reset_index(name='Total')
# print('df2',df_2.head(2))

df_Res = pd.merge(df_1, df_2, on='ClaseReal', how='left')

# Crea la columna resultado
df_Res['Porcentaje'] = ((df_Res['Aciertos'] / df_Res['Total']) * 100).__round__(2)

# print(df_Res)

# Ahora hacer una grafica que ponga la ClaseReal respecto al Porcentaje
# sns.barplot(df_Res, x=df_Res['ClaseReal'], y=df_Res['Porcentaje'])
# plt.show()

# Eso es demasiado grande, asi que mejor hacer la grafica con los que tienen mas porcentaje
# mayores = df_Res[df_Res['Porcentaje'] > 70]
# sns.barplot(mayores, x='ClaseReal', y='Porcentaje')
# plt.show()

'''
print("################################################################################################################")
# Algoritmos que manejamos:  ['Ecualización del histograma (EH)'
#  'Ecualización del histograma adaptativa limitada por contraste (EHALC)'
#  'Corrección de gamma (CG)' 'Transformación logarítmica (TL)']
print("################################################################################################################")

# Aciertos para el algoritmo Ecualizacion del histograma con ClasePred
#df_EH = df[(df['ClaseReal'] == df['ClasePred']) & (df['Algoritmo'] == 'Ecualización del histograma')]
df_EH = df_filtered[df_filtered['Algoritmo'] == 'Ecualización del histograma']
df_EH = df_EH.groupby('ClaseReal')['ClasePred'].count().reset_index(name='EH')

# Representa como de importante es cada clase respecto a ese algoritmo
df_EH['Porcentaje'] = ((df_EH['EH'] / df_EH['EH'].sum()) * 100).__round__(3)
df_EH = df_EH.sort_values(by='EH', ascending=False)

# Mostrar el DataFrame resultante
print(df_EH.head(5))
# print('Aciertos totales', df_EH['EH'].sum())
# print('TOTAL', df_EH['Porcentaje'].sum())

# Algunas graficas que pueden servir, pero hay que buscarle algun uso
# sns.barplot(df_EH[df_EH['Porcentaje'] > 1.5], x='ClaseReal', y='EH')
# sns.scatterplot(df_EH, x='ClaseReal', y='EH')
# sns.scatterplot(df_EH, x='EH', y='Porcentaje')
# plt.show()

# TOP 10 con EH
sns.barplot(df_EH.head(10), x='ClaseReal', y='EH')
plt.title('Top 10 Clases Más Predichas con EH')
plt.xlabel('Clase')
plt.xlabel('Cantidad')
plt.show()
'''

'''
print("################################################################################################################")
print("################################################################################################################")

df_EHALC = df_filtered[df_filtered['Algoritmo'] == 'Ecualización del histograma adaptativa limitada por contraste']
df_EHALC = df_EHALC.groupby('ClaseReal')['ClasePred'].count().reset_index(name='EHALC')

# Representa como de importante es cada clase respecto a ese algoritmo
df_EHALC['Porcentaje'] = ((df_EHALC['EHALC'] / df_EHALC['EHALC'].sum()) * 100).__round__(3)
df_EHALC = df_EHALC.sort_values(by='EHALC', ascending=False)

# Mostrar el DataFrame resultante
print(df_EHALC.head(5))
print('Aciertos totales', df_EHALC['EHALC'].sum())
print('TOTAL', df_EHALC['Porcentaje'].sum())

# TOP 10 con EHALC
sns.barplot(df_EHALC.head(10), x='ClaseReal', y='EHALC')
plt.title('Top 10 Clases Más Predichas con EHALC')
plt.xlabel('Clase')
plt.xlabel('Cantidad')
plt.show()
'''
print("################################################################################################################")
print("################################################################################################################")
# 'Corrección de gamma (CG)'
df_filtrado = df[df['Algoritmo'] == 'Corrección de gamma']
df_CG = df_filtrado[df_filtrado['ClaseReal'] == df_filtrado['ClasePred']]
print(df_CG)

df_CG = df_CG.groupby('ClaseReal')['ClasePred'].count().reset_index(name='CG')

# Representa como de importante es cada clase respecto a ese algoritmo
df_CG['Porcentaje'] = ((df_CG['CG'] / df_CG['CG'].sum()) * 100).__round__(3)
df_CG = df_CG.sort_values(by='CG', ascending=False)

# Mostrar el DataFrame resultante
print(df_CG.head(5))
print('Aciertos totales', df_CG['CG'].sum())
print('TOTAL', df_CG['Porcentaje'].sum())

# TOP 10 con EHALC
sns.barplot(df_CG.head(10), x='ClaseReal', y='CG')
plt.title('Top 10 Clases Más Predichas con CG')
plt.xlabel('Clase')
plt.xlabel('Cantidad')
plt.show()
