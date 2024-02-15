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

print("################################################################################################################")

# CLASE REAL RESPECTO A CLASE PREDICHA PRIME ya que es la que interviene en la mejora de la imagen, la clasePred normal solo es clasificacion, por eso da todas igual

# Filtrar las filas donde ClaseReal y ClasePredPrime son iguales
coincidencias = df[df['ClaseReal'] == df['ClasePredPrime']]
# print(coincidencias)

# Agrupar las que son iguales y contarlas
df_1 = coincidencias.groupby('ClaseReal')['ClasePredPrime'].count().reset_index(name='Hit')
# print('df1',df_1.head(2))

df_2 = df.groupby('ClaseReal')['ClasePredPrime'].count().reset_index(name='Total')
# print('df2',df_2.head(2))

df_Res = pd.merge(df_1, df_2, on='ClaseReal', how='left')

# Crea la columna resultado
df_Res['Porcentaje'] = ((df_Res['Hit'] / df_Res['Total']) * 100).__round__(2)

# print(df_Res)

# Ahora hacer una grafica que ponga la ClaseReal respecto al Porcentaje
# sns.barplot(df_Res, x=df_Res['ClaseReal'], y=df_Res['Porcentaje'])
# plt.show()

# Eso es demasiado grande, asi que mejor hacer la grafica con los que tienen mas porcentaje
# mayores = df_Res[df_Res['Porcentaje'] > 50]
# sns.barplot(mayores.head(10), x='ClaseReal', y='Porcentaje')
# plt.title('Top 10 Clases Más Predichas (GLOBAL)')
# plt.show()

'''
print("################################################################################################################")
# Algoritmos que manejamos:  ['Ecualización del histograma (EH)'
#  'Ecualización del histograma adaptativa limitada por contraste (EHALC)'
#  'Corrección de gamma (CG)' 'Transformación logarítmica (TL)']
print("################################################################################################################")
'''

'''
# Aciertos para el algoritmo Ecualizacion del histograma con ClasePredPrime
df_EH = coincidencias[coincidencias['Algoritmo'] == 'Ecualización del histograma']
df_EH = df_EH.groupby('ClaseReal')['ClasePredPrime'].count().reset_index(name='Hit')

# Representa como de importante es cada clase respecto a ese algoritmo
df_EH['Porcentaje'] = ((df_EH['Hit'] / df_EH['Hit'].sum()) * 100).__round__(3)
df_EH = df_EH.sort_values(by='Hit', ascending=False)

# Mostrar el DataFrame resultante
print(df_EH.head(5))
print('Aciertos totales', df_EH['Hit'].sum())
print('TOTAL', df_EH['Porcentaje'].sum())
'''

# Solo sirve la primera grafica, la otra no le veo mucho sentido
# sns.barplot(df_EH[df_EH['Hit'] > 10], x='ClaseReal', y='Hit')
# sns.scatterplot(df_EH, x='ClaseReal', y='Hit')
# plt.show()

'''
# TOP 10 con EH -- Falta recortar, es decir, que la grafica empiece con el valor minimo
sns.barplot(df_EH.head(10), x='ClaseReal', y='Hit')
plt.title('Top 10 Clases Más Predichas con EH')
plt.xlabel('Clase')
plt.xlabel('Cantidad')
plt.show()
'''

'''
print("################################ ")
# Grafico circular, falta perfilar la explicacion de realmente lo que sucede aqui
df_circularEH = df_EH.groupby('Hit').count().reset_index()
df_circularEH.drop(df_circularEH.columns[2], axis=1, inplace=True)
print(df_circularEH)

fig, ax = plt.subplots()
ax.pie(df_circularEH['ClaseReal'], labels=df_circularEH['Hit'], autopct='%1.1f%%')
plt.title('Porcentaje de predicciones con EH')
plt.show()
'''

print("################################################################################################################")
print("################################################################################################################")

# Aciertos para el algoritmo Ecualizacion del histograma adaptativa limitada por contraste
df_EHALC = coincidencias[coincidencias['Algoritmo'] == 'Ecualización del histograma adaptativa limitada por contraste']
df_EHALC = df_EHALC.groupby('ClaseReal')['ClasePredPrime'].count().reset_index(name='Hit')

# Representa como de importante es cada clase respecto a ese algoritmo
df_EHALC['Porcentaje'] = ((df_EHALC['Hit'] / df_EHALC['Hit'].sum()) * 100).__round__(3)
df_EHALC = df_EHALC.sort_values(by='Hit', ascending=False)

# Mostrar el DataFrame resultante
print(df_EHALC.head(5))
print('Aciertos totales', df_EHALC['Hit'].sum())
print('TOTAL', df_EHALC['Porcentaje'].sum())


# TOP 10 con EHALC
sns.barplot(df_EHALC.head(10), x='ClaseReal', y='Hit')
plt.title('Top 10 Clases Más Predichas con EHALC')
plt.xlabel('Clase')
plt.xlabel('Cantidad')
plt.show()


'''
sns.barplot(df_EHALC.tail(10), x='ClaseReal', y='EHALC')
plt.title('Top 10 Clases Menos Predichas con EHALC')
plt.xlabel('Clase')
plt.xlabel('Cantidad')
plt.show()
'''

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

# Algunas graficas que pueden servir, pero hay que buscarle algun uso
# sns.barplot(df_CG[df_CG['Porcentaje'] > 1.5], x='ClaseReal', y='CG')
# sns.scatterplot(df_CG, x='ClaseReal', y='CG')
# sns.scatterplot(df_CG, x='CG', y='Porcentaje')
# plt.show()

'''

'''
# TOP 10 con GC
sns.barplot(df_CG.head(10), x='ClaseReal', y='CG')
plt.title('Top 10 Clases Más Predichas con CG')
plt.xlabel('Clase')
plt.xlabel('Cantidad')
plt.show()
'''

'''
sns.barplot(df_CG.tail(10), x='ClaseReal', y='CG')
plt.title('Top 10 Clases Menos Predichas con CG')
plt.xlabel('Clase')
plt.xlabel('Cantidad')
plt.show()
'''

