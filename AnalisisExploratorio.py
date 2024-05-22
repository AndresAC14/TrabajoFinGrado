# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from wordcloud import WordCloud
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 350)

# Importar el dataframe
df = pd.read_csv('Conjunto_Entrenamiento_10000.csv')
jerarquia = pd.read_csv('Jerarquia_Clases.csv')

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
# JOIN de DF grande con el de las jerarquias por 'ClaseReal'
df['ClaseReal'] = df['ClaseReal'].str.lower()
df['ClasePred'] = df['ClasePred'].str.lower()
df['ClasePredPrime'] = df['ClasePredPrime'].str.lower()
df = pd.merge(df, jerarquia, on='ClaseReal', how='left')
df['Nivel'] = df['Nivel'].astype('Int64')

# Ejemplo mostrar todos los persian cat
persian = df[df['ClaseReal'] == 'persian cat']
print(persian)


print("################################################################################################################")

print('Porcentaje acierto total (Todos algoritmos)')
# CLASE REAL RESPECTO A CLASE PREDICHA PRIME ya que es la que interviene en la mejora de la imagen

# Filtrar las filas donde ClaseReal y ClasePredPrime son iguales
coincidencias = df[df['ClaseReal'] == df['ClasePredPrime']]
# print(coincidencias)

# Agrupar las que son iguales y contarlas
df_1 = coincidencias.groupby('ClaseReal')['ClasePredPrime'].count().reset_index(name='Hit')
# print('df1',df_1.head(2))

df_2 = df.groupby('ClaseReal')['ClasePredPrime'].count().reset_index(name='Total')
# print('df2',df_2.head(2))

df_clases = pd.merge(df_1, df_2, on='ClaseReal', how='left')

# Crea la columna resultado
df_clases['Porcentaje'] = ((df_clases['Hit'] / df_clases['Total']) * 100).__round__(3)
df_clases = df_clases.sort_values(by='Porcentaje', ascending=False)
print(df_clases.head(5))

# Ahora hacer una grafica que ponga la ClaseReal respecto al Porcentaje
# sns.barplot(df_clases.head(10), x='ClaseReal', y='Porcentaje')
# plt.title('Top 10 Clases Más Predichas (GLOBAL)')
# plt.show()


print("################################################################################################################")
# Algoritmos que manejamos:  ['Ecualización del histograma (EH)'
#  'Ecualización del histograma adaptativa limitada por contraste (EHALC)'
#  'Corrección de gamma (CG)' 'Transformación logarítmica (TL)']
print("################################################################################################################")

# Aciertos para el algoritmo Ecualizacion del histograma con ClasePredPrime
df_EH = coincidencias[coincidencias['Algoritmo'] == 'Ecualización del histograma']
df_EH = df_EH.groupby('ClaseReal')['ClasePredPrime'].count().reset_index(name='Hit')

# Representa como de importante es cada clase respecto a ese algoritmo
df_EH['Porcentaje'] = ((df_EH['Hit'] / df_EH['Hit'].sum()) * 100).__round__(3)
df_EH = df_EH.sort_values(by='Hit', ascending=False)
aciertos_EH = df_EH['Hit'].sum()

# Mostrar el DataFrame resultante
print(df_EH.head(5))
print('Aciertos totales EH', aciertos_EH)

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
# Grafico circular
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
aciertos_EHALC = df_EHALC['Hit'].sum()

# Mostrar el DataFrame resultante
print(df_EHALC.head(5))
print('Aciertos totales EHALC', aciertos_EHALC)


'''
# TOP 10 con EHALC
sns.barplot(df_EHALC.head(10), x='ClaseReal', y='Hit')
plt.title('Top 10 Clases Más Predichas con EHALC')
plt.xlabel('Clase')
plt.xlabel('Cantidad')
plt.show()
'''

'''
sns.barplot(df_EHALC.tail(10), x='ClaseReal', y='EHALC')
plt.title('Top 10 Clases Menos Predichas con EHALC')
plt.xlabel('Clase')
plt.xlabel('Cantidad')
plt.show()
'''


print("################################################################################################################")
print("################################################################################################################")

# 'Corrección de gamma (CG)'
df_CG = coincidencias[coincidencias['Algoritmo'] == 'Corrección de gamma']
df_CG = df_CG.groupby('ClaseReal')['ClasePredPrime'].count().reset_index(name='Hit')

# Representa como de importante es cada clase respecto a ese algoritmo
df_CG['Porcentaje'] = ((df_CG['Hit'] / df_CG['Hit'].sum()) * 100).__round__(3)
df_CG = df_CG.sort_values(by='Hit', ascending=False)
aciertos_CG = df_CG['Hit'].sum()

# Mostrar el DataFrame resultante
print(df_CG.head(5))
print('Aciertos totales CG', aciertos_CG)


'''
# TOP 10 con GC
sns.barplot(df_CG.head(10), x='ClaseReal', y='Hit')
plt.title('Top 10 Clases Más Predichas con CG')
plt.xlabel('Clase')
plt.xlabel('Cantidad')
plt.show()
'''

'''
sns.barplot(df_CG.tail(10), x='ClaseReal', y='Hit')
plt.title('Top 10 Clases Menos Predichas con CG')
plt.xlabel('Clase')
plt.xlabel('Cantidad')
plt.show()
'''


print("################################################################################################################")
print("################################################################################################################")

# 'Transformación logarítmica (TL)'
df_TL = coincidencias[coincidencias['Algoritmo'] == 'Transformación logarítmica']
df_TL = df_TL.groupby('ClaseReal')['ClasePredPrime'].count().reset_index(name='Hit')

# Representa como de importante es cada clase respecto a ese algoritmo
df_TL['Porcentaje'] = ((df_TL['Hit'] / df_TL['Hit'].sum()) * 100).__round__(3)
df_TL = df_TL.sort_values(by='Hit', ascending=False)
aciertos_TL = df_TL['Hit'].sum()

# Mostrar el DataFrame resultante
print(df_TL.head(5))
print('Aciertos totales TL', aciertos_TL)


'''
# TOP 10 con TL
sns.barplot(df_TL.head(10), x='ClaseReal', y='Hit')
plt.title('Top 10 Clases Más Predichas con TL')
plt.xlabel('Clase')
plt.xlabel('Cantidad')
plt.show()
'''


print("################################################################################################################")
print("################################################################################################################")

# Algoritmo con más aciertos
col_aciertos = ['EH', 'EHALC', 'CG', 'TL']
val_aciertos = [aciertos_EH, aciertos_EHALC, aciertos_CG, aciertos_TL]
df_aciertos = pd.DataFrame([val_aciertos], columns=col_aciertos, index=['Aciertos'])

print(df_aciertos)

'''
# Gráfico circular que muestra el porcentaje de acierto por algoritmo
plt.figure(figsize=(8, 8))
plt.pie(df_aciertos.iloc[0].values, labels=df_aciertos.columns, autopct='%1.1f%%', startangle=90)
plt.title('Aciertos por Algoritmo')
plt.show()
'''

print("################################################################################################################")
print("################################################################################################################")

print('Ecualizacion del histograma (EH)')
# Ver cuanto mejora la imagen/clase con Ecualizacion del histograma
mejora_EH = df[df['Algoritmo'] == 'Ecualización del histograma']
mejora_EH = mejora_EH.drop(['MediaPrime', 'Mediana', 'DesvTipica', 'MedianaPrime', 'DesvTipicaPrime', 'Algoritmo', 'ClasePred'], axis=1)

# Porcentaje de mejora de probabilidad de ser predicha correctamente = ((nuevo - antiguo) / antiguo) * 100 )
mejora_EH['% Mejora'] = (((mejora_EH['ProbClasePredPrime'] - mejora_EH['ProbClasePred']) / mejora_EH['ProbClasePred']) * 100).__round__(3)
mejora_EH['¿Cambia?'] = mejora_EH.apply(lambda fila: 0 if fila['ClaseReal'] == fila['ClasePredPrime'] else 1, axis=1)
print(mejora_EH.head(10))
print(mejora_EH['¿Cambia?'].value_counts())


'''
# Este grafico compara en una nube de puntos el % de mejora respecto de la iluminacion (media original de la imagen)
# Crear una figura y un eje
fig, ax = plt.subplots()
# Asignar colores según los valores de '¿Cambia?'
colores = ['green' if valor == 0 else 'red' for valor in mejora_EH['¿Cambia?']]
# Crear el gráfico de dispersión
scatter = ax.scatter(mejora_EH['Media'], mejora_EH['% Mejora'], c=colores)
# Etiquetas y título
ax.set_xlabel('Media Original')
ax.set_ylabel('% Mejora')
ax.set_title('Mejora Con EH')
# Crear manualmente las leyendas
legend_labels = [mpatches.Patch(color='green', label='Si'), mpatches.Patch(color='red', label='No')]
ax.legend(handles=legend_labels, title='¿Cambia?')
# Mostrar el gráfico
plt.show()
'''

'''
# ¿Qué clase mejora más? -> Es decir, que sea más probable de ser predicha
# Se hace la media de todos los elementos que tengan la misma clase
clase_EH = mejora_EH.groupby('ClaseReal')['% Mejora'].mean().__round__(3).reset_index(name='% Mejora Por Clase')
clase_EH = clase_EH.sort_values(by='% Mejora Por Clase', ascending=False)
#print('Top 10 mejores clases')
#print(clase_EH.head(10))


# Grafico 10 clases mas mejoradas con TL
sns.barplot(clase_EH.head(10), x='ClaseReal', y='% Mejora Por Clase')
plt.xlabel('Clase Real')
plt.ylabel('% Mejora Media Por Clase')
plt.title('Top 10 Clases Más Mejoradas Con EH')
# Rotar las etiquetas
plt.xticks(rotation=45, ha='right')
# Mostrar el gráfico
plt.show()
'''

print("################################################################################################################")
print("################################################################################################################")

print('Ecualización del histograma adaptativa limitada por contraste (EHALC)')
# Ver cuanto mejora la imagen/clase con 'Ecualización del histograma adaptativa limitada por contraste'
mejora_EHALC = df[df['Algoritmo'] == 'Ecualización del histograma adaptativa limitada por contraste']
mejora_EHALC = mejora_EHALC.drop(['MediaPrime', 'Mediana', 'DesvTipica', 'MedianaPrime', 'DesvTipicaPrime', 'Algoritmo', 'ClasePred'], axis=1)

# Porcentaje de mejora = ((nuevo - antiguo) / antiguo) * 100 )
mejora_EHALC['% Mejora'] = (((mejora_EHALC['ProbClasePredPrime'] - mejora_EHALC['ProbClasePred']) / mejora_EHALC['ProbClasePred']) * 100).__round__(3)
mejora_EHALC['¿Cambia?'] = mejora_EHALC.apply(lambda fila: 0 if fila['ClaseReal'] == fila['ClasePredPrime'] else 1, axis=1)
print(mejora_EHALC.head(10))
print(mejora_EHALC['¿Cambia?'].value_counts())


'''
# Este grafico compara en una nube de puntos el % de mejora respecto de la iluminacion (media original de la imagen)
# Crear una figura y un eje
fig, ax = plt.subplots()
# Asignar colores según los valores de '¿Cambia?'
colores = ['green' if valor == 1 else 'red' for valor in mejora_EHALC['¿Cambia?']]
# Crear el gráfico de dispersión
scatter = ax.scatter(mejora_EHALC['Media'], mejora_EHALC['% Mejora'], c=colores)
# Etiquetas y título
ax.set_xlabel('Media Original')
ax.set_ylabel('% Mejora')
ax.set_title('Mejora Con EHALC')
# Crear manualmente las leyendas
legend_labels = [mpatches.Patch(color='green', label='Si'), mpatches.Patch(color='red', label='No')]
ax.legend(handles=legend_labels, title='¿Cambia?')
# Mostrar el gráfico
plt.show()
'''

'''
# ¿Qué clase mejora más? -> Es decir, que sea más probable de ser predicha
# Se hace la media de todos los elementos que tengan la misma clase
clase_EHALC = mejora_EHALC.groupby('ClaseReal')['% Mejora'].mean().__round__(3).reset_index(name='% Mejora Por Clase')
clase_EHALC = clase_EHALC.sort_values(by='% Mejora Por Clase', ascending=False)
print('Top 10 mejores clases')
print(clase_EHALC.head(10))

# Grafico 10 clases mas mejoradas con TL
sns.barplot(clase_EHALC.head(10), x='ClaseReal', y='% Mejora Por Clase')
plt.xlabel('Clase Real')
plt.ylabel('% Mejora Media Por Clase')
plt.title('Top 10 Clases Más Mejoradas Con EHALC')
# Rotar las etiquetas
plt.xticks(rotation=45, ha='right')
# Mostrar el gráfico
plt.show()
'''

print("################################################################################################################")
print("################################################################################################################")

print('Corrección de gamma (CG)')
# Ver cuanto mejora la imagen/clase con 'Corrección de gamma (CG)'
mejora_CG = df[df['Algoritmo'] == 'Corrección de gamma']
mejora_CG = mejora_CG.drop(['MediaPrime', 'Mediana', 'DesvTipica', 'MedianaPrime', 'DesvTipicaPrime', 'Algoritmo', 'ClasePred'], axis=1)

# Porcentaje de mejora = ((nuevo - antiguo) / antiguo) * 100 )
mejora_CG['% Mejora'] = (((mejora_CG['ProbClasePredPrime'] - mejora_CG['ProbClasePred']) / mejora_CG['ProbClasePred']) * 100).__round__(3)
mejora_CG['¿Cambia?'] = mejora_CG.apply(lambda fila: 0 if fila['ClaseReal'] == fila['ClasePredPrime'] else 1, axis=1)
print(mejora_CG.head(10))
print(mejora_CG['¿Cambia?'].value_counts())


'''
# Este grafico compara en una nube de puntos el % de mejora respecto de la iluminacion (media original de la imagen)
# Crear una figura y un eje
fig, ax = plt.subplots()
# Asignar colores según los valores de '¿Cambia?'
colores = ['green' if valor == 1 else 'red' for valor in mejora_CG['¿Cambia?']]
# Crear el gráfico de dispersión
scatter = ax.scatter(mejora_CG['Media'], mejora_CG['% Mejora'], c=colores)
# Etiquetas y título
ax.set_xlabel('Media Original')
ax.set_ylabel('% Mejora')
ax.set_title('Mejora Con CG')
# Crear manualmente las leyendas
legend_labels = [mpatches.Patch(color='green', label='Si'), mpatches.Patch(color='red', label='No')]
ax.legend(handles=legend_labels, title='¿Cambia?')
# Mostrar el gráfico
plt.show()
'''

'''
# ¿Qué clase mejora más? -> Es decir, que sea más probable de ser predicha
# Se hace la media de todos los elementos que tengan la misma clase
clase_CG = mejora_CG.groupby('ClaseReal')['% Mejora'].mean().__round__(3).reset_index(name='% Mejora Por Clase')
clase_CG = clase_CG.sort_values(by='% Mejora Por Clase', ascending=False)
print('Top 10 mejores clases')
print(clase_CG.head(10))

# Grafico 10 clases mas mejoradas con TL
sns.barplot(clase_CG.head(10), x='ClaseReal', y='% Mejora Por Clase')
plt.xlabel('Clase Real')
plt.ylabel('% Mejora Media Por Clase')
plt.title('Top 10 Clases Más Mejoradas Con CG')
# Rotar las etiquetas
plt.xticks(rotation=45, ha='right')
# Mostrar el gráfico
plt.show()
'''

print("################################################################################################################")
print("################################################################################################################")

print('Transformación logarítmica (TL)')
# Ver cuanto mejora la imagen/clase con 'Transformación logarítmica (TL)'
mejora_TL = df[df['Algoritmo'] == 'Transformación logarítmica']
mejora_TL = mejora_TL.drop(['MediaPrime', 'Mediana', 'DesvTipica', 'MedianaPrime', 'DesvTipicaPrime', 'Algoritmo', 'ClasePred'], axis=1)

# Porcentaje de mejora = ((nuevo - antiguo) / antiguo) * 100 )
mejora_TL['% Mejora'] = (((mejora_TL['ProbClasePredPrime'] - mejora_TL['ProbClasePred']) / mejora_TL['ProbClasePred']) * 100).__round__(3)
mejora_TL['¿Cambia?'] = mejora_TL.apply(lambda fila: 0 if fila['ClaseReal'] == fila['ClasePredPrime'] else 1, axis=1)
print(mejora_TL.head(10))
print(mejora_TL['¿Cambia?'].value_counts())

'''
# Este grafico compara en una nube de puntos el % de mejora respecto de la iluminacion (media original de la imagen)
# Crear una figura y un eje
fig, ax = plt.subplots()
# Asignar colores según los valores de '¿Cambia?'
colores = ['green' if valor == 1 else 'red' for valor in mejora_TL['¿Cambia?']]
# Crear el gráfico de dispersión
scatter = ax.scatter(mejora_TL['Media'], mejora_TL['% Mejora'], c=colores)
# Etiquetas y título
ax.set_xlabel('Media Original')
ax.set_ylabel('% Mejora')
ax.set_title('Mejora Con TL')
# Crear manualmente las leyendas
legend_labels = [mpatches.Patch(color='green', label='Si'), mpatches.Patch(color='red', label='No')]
ax.legend(handles=legend_labels, title='¿Cambia?')
# Mostrar el gráfico
plt.show()
'''

'''
# ¿Qué clase mejora más? -> Es decir, que sea más probable de ser predicha
# Se hace la media de todos los elementos que tengan la misma clase
clase_TL = mejora_TL.groupby('ClaseReal')['% Mejora'].mean().__round__(3).reset_index(name='% Mejora Por Clase')
clase_TL = clase_TL.sort_values(by='% Mejora Por Clase', ascending=False)
print('Top 10 mejores clases')
print(clase_TL.head(10))

# Grafico 10 clases mas mejoradas con TL
sns.barplot(clase_TL.head(10), x='ClaseReal', y='% Mejora Por Clase')
plt.xlabel('Clase Real')
plt.ylabel('% Mejora Media Por Clase')
plt.title('Top 10 Clases Más Mejoradas Con TL')
# Rotar las etiquetas
plt.xticks(rotation=45, ha='right')
# Mostrar el gráfico
plt.show()
'''

print("################################################################################################################")
print("################################################################################################################")

# Que clase es mas predicha segun el nivel
niveles = df['Nivel'].value_counts().reset_index('Nivel')
niveles = niveles.sort_values(by='Nivel', ascending=True)

print(niveles)

'''
# Grafico circular de los niveles y el count por nivel -> REVISAR PARA EL TAM DE LOS %
fig, ax = plt.subplots(figsize=(10, 10))
ax.pie(niveles['count'], labels=niveles['Nivel'], autopct='%1.1f%%', startangle=0,
                                  textprops={'fontsize': 10}, pctdistance=0.8)
plt.title('Cantidad de Clases Por Nivel')
plt.show()
'''

print("################################################################################################################")
print("################################################################################################################")

# Comparacion Mejora por Nivel con Algoritmos de Mejora de Imagen
def comparacion_niveles(algoritmo):

    niveles_mejora = []
    df_mejora = pd.DataFrame
    top_clases_nivel = pd.DataFrame(columns=['Nivel', 'ClaseReal', '% Mejora Por Clase'])

    if algoritmo == 'EH':
        df_mejora = mejora_EH

    elif algoritmo == 'EHALC':
        df_mejora = mejora_EHALC

    elif algoritmo == 'TL':
        df_mejora = mejora_TL

    elif algoritmo == 'CG':
        df_mejora = mejora_CG

    for i in niveles['Nivel'].values:
        nodos = df_mejora[df_mejora['Nivel'] == i]
        mejora = nodos['% Mejora'].mean().__round__(2)
        niveles_mejora.append((i, mejora))

        clases = nodos.groupby('ClaseReal')['% Mejora'].mean().__round__(2).reset_index(name='% Mejora Por Clase')
        clases = clases.sort_values(by='% Mejora Por Clase', ascending=False).head(5)

        # Añadir el nivel a las clases para el DataFrame final
        clases['Nivel'] = i

        # Append al DataFrame top_clases_nivel
        top_clases_nivel = pd.concat([top_clases_nivel, clases])

    return niveles_mejora, top_clases_nivel
    # print(top_clases_nivel)


niveles_mejora_EH, topClases_EH = comparacion_niveles('EH')
'''
df_mejoras_EH = pd.DataFrame(niveles_mejora_EH, columns=['Nivel', 'Mejora'])

plt.figure(figsize=(10, 10))
plt.plot(df_mejoras_EH['Nivel'], df_mejoras_EH['Mejora'], marker='o', linestyle='-', color='b')
plt.title('Mejora por Nivel con Ecualización del Histograma')
plt.xlabel('Nivel')
plt.ylabel('% Mejora')
plt.xticks(niveles['Nivel'].values)
plt.grid(True)
plt.show()
'''

print(topClases_EH)

# Definir colores para cada nivel del 1 al 13
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#ff5733', '#7a33ff', '#33ff99']


# Crear el gráfico con seaborn
plt.figure(figsize=(15, 8))
ax = sns.barplot(data=topClases_EH, x='Nivel', y='% Mejora Por Clase', hue='ClaseReal', dodge=True)

# Añadir etiquetas y título
plt.xlabel('Nivel')
plt.ylabel('% Mejora')
ax.get_legend().remove()
plt.title('% Mejora por Nivel y ClaseReal')


# Añadir nombres de ClaseReal en las barras
for p, (_, row) in zip(ax.patches, topClases_EH.iterrows()):
    height = p.get_height()
    ax.annotate(
        f'{row["ClaseReal"]}',  # Nombre de ClaseReal
        (p.get_x() + p.get_width() / 2., height),
        ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 10),  # Desplazamiento vertical
        textcoords='offset points'
    )
'''
# Añadir texto de ClaseReal dentro de cada barra
for i, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(
        p.get_x() + p.get_width() / 2,  # x-coordinate
        height + 0.5,  # y-coordinate, ajustando un poco más arriba
        topClases_EH.iloc[i % len(topClases_EH)]["ClaseReal"],  # texto de ClaseReal
        ha='center',  # horizontal alignment
        va='bottom'  # vertical alignment
    )
'''
# Ajustar los márgenes para que la leyenda no se corte
#plt.tight_layout()

# Mostrar el gráfico
plt.show()

print("################################################################################################################")
print("################################################################################################################")
