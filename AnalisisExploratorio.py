# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 400)

# Importar el dataframe
df = pd.read_csv('Conjunto_Entrenamiento_10000.csv')
jerarquia = pd.read_csv('Jerarquia_Clases.csv')
jerarquiaPred = pd.read_csv('Jerarquia_Clases_Pred.csv')

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
sns.barplot(df['ClaseReal'].value_counts().head(10))
plt.title('Clase Real más común')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45, ha='right')
plt.show()


# ClasePred mas comun grafico
sns.barplot(df['ClasePred'].value_counts().head(10))
plt.title('Clase Predicha más común')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45, ha='right')
plt.show()


# ClasePredPrime mas comun grafico
sns.barplot(df['ClasePredPrime'].value_counts().head(10))
plt.title('Clase Predicha Prime más común')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45, ha='right')
plt.show()
'''

print(df.sample(10))

print("################################################################################################################")
# JOIN de DF grande con el de las jerarquias por 'ClaseReal'
df['ClaseReal'] = df['ClaseReal'].str.lower()
df['ClasePred'] = df['ClasePred'].str.lower()
df['ClasePredPrime'] = df['ClasePredPrime'].str.lower()
df = pd.merge(df, jerarquia, on='ClaseReal', how='left')
df['Nivel'] = df['Nivel'].astype('Int64')

# Merge de los Pred
df = pd.merge(df, jerarquiaPred, on='ClasePredPrime', how='left')

# Funcion de aciertos Real Padre y Abuelo
# Añadimos las columnas de Aciertos
df['Aciertos_Real'] = (df['ClaseReal'] == df['ClasePredPrime']).astype(int)
df['Aciertos_Padre'] = (df['ClasePadre'] == df['ClasePredPadre']).astype(int)
df['Aciertos_Abuelo'] = (df['ClaseAbuelo'] == df['ClasePredAbuelo']).astype(int)

# La que falta que seria
# Crear la columna de aciertos totales
df['Aciertos_Totales'] = ((df['ClaseReal'] != df['ClasePredPrime']) & ((df['ClasePadre'] == df['ClasePredPadre']) | (df['ClaseAbuelo'] == df['ClasePredAbuelo']))).astype(int)


# print(df.head(15))

print(df.sample(10))

'''
# Ejemplo mostrar todos los persian cat
persian = df[df['ClaseReal'] == 'persian cat']
print(persian)
'''

print("################################################################################################################")

print('Porcentaje acierto total (Todos algoritmos)')
# CLASE REAL RESPECTO A CLASE PREDICHA PRIME ya que es la que interviene en la mejora de la imagen

# Filtrar las filas donde ClaseReal y ClasePredPrime son iguales
coincidencias = df[df['ClaseReal'] == df['ClasePredPrime']]
# print(coincidencias)

# Agrupar las que son iguales y contarlas
df_1 = coincidencias.groupby('ClaseReal')['ClasePredPrime'].count().reset_index(name='Aciertos')
# print('df1',df_1.head(2))

df_2 = df.groupby('ClaseReal')['ClasePredPrime'].count().reset_index(name='Total')
# print('df2',df_2.head(2))

df_clases = pd.merge(df_1, df_2, on='ClaseReal', how='left')

# Crea la columna resultado
df_clases['Porcentaje'] = ((df_clases['Aciertos'] / df_clases['Total']) * 100).__round__(3)
df_clases = df_clases.sort_values(by='Porcentaje', ascending=False)
print(df_clases.head(5))

'''
# Ahora hacer una grafica que ponga la ClaseReal respecto al Porcentaje
sns.barplot(df_clases.head(10), x='ClaseReal', y='Porcentaje')
plt.title('Top 10 Clases Más Predichas (GLOBAL)')
plt.xlabel('Clase')
plt.xticks(rotation=45, ha='right')
plt.show()
'''

print("################################################################################################################")
# Algoritmos que manejamos:  ['Ecualización del histograma (EH)'
#  'Ecualización del histograma adaptativa limitada por contraste (EHALC)'
#  'Corrección de gamma (CG)' 'Transformación logarítmica (TL)']
print("################################################################################################################")

# Aciertos para el algoritmo Ecualizacion del histograma con ClasePredPrime
df_EH = coincidencias[coincidencias['Algoritmo'] == 'Ecualización del histograma']
df_EH = df_EH.groupby('ClaseReal')['ClasePredPrime'].count().reset_index(name='Aciertos')

# Representa como de importante es cada clase respecto a ese algoritmo
df_EH['Porcentaje'] = ((df_EH['Aciertos'] / df_EH['Aciertos'].sum()) * 100).__round__(3)
df_EH = df_EH.sort_values(by='Aciertos', ascending=False)
aciertos_EH = df_EH['Aciertos'].sum()

# Mostrar el DataFrame resultante
print(df_EH.head(5))
print('Aciertos totales EH', aciertos_EH)

'''
# TOP 10 con EH 
sns.barplot(df_EH.head(10), x='ClaseReal', y='Aciertos')
plt.title('Top 10 Clases Más Predichas con EH')
plt.xlabel('Clase')
plt.ylabel('Aciertos')
plt.xticks(rotation=45, ha='right')
plt.show()
'''

print("################################################################################################################")
print("################################################################################################################")

# Aciertos para el algoritmo Ecualizacion del histograma adaptativa limitada por contraste
df_EHALC = coincidencias[coincidencias['Algoritmo'] == 'Ecualización del histograma adaptativa limitada por contraste']
df_EHALC = df_EHALC.groupby('ClaseReal')['ClasePredPrime'].count().reset_index(name='Aciertos')

# Representa como de importante es cada clase respecto a ese algoritmo
df_EHALC['Porcentaje'] = ((df_EHALC['Aciertos'] / df_EHALC['Aciertos'].sum()) * 100).__round__(3)
df_EHALC = df_EHALC.sort_values(by='Aciertos', ascending=False)
aciertos_EHALC = df_EHALC['Aciertos'].sum()

# Mostrar el DataFrame resultante
print(df_EHALC.head(5))
print('Aciertos totales EHALC', aciertos_EHALC)


'''
# TOP 10 con EHALC
sns.barplot(df_EHALC.head(10), x='ClaseReal', y='Aciertos')
plt.title('Top 10 Clases Más Predichas con EHALC')
plt.xlabel('Clase')
plt.ylabel('Aciertos')
plt.xticks(rotation=45, ha='right')
plt.show()
'''


print("################################################################################################################")
print("################################################################################################################")

# 'Corrección de gamma (CG)'
df_CG = coincidencias[coincidencias['Algoritmo'] == 'Corrección de gamma']
df_CG = df_CG.groupby('ClaseReal')['ClasePredPrime'].count().reset_index(name='Aciertos')

# Representa como de importante es cada clase respecto a ese algoritmo
df_CG['Porcentaje'] = ((df_CG['Aciertos'] / df_CG['Aciertos'].sum()) * 100).__round__(3)
df_CG = df_CG.sort_values(by='Aciertos', ascending=False)
aciertos_CG = df_CG['Aciertos'].sum()

# Mostrar el DataFrame resultante
print(df_CG.head(5))
print('Aciertos totales CG', aciertos_CG)


'''
# TOP 10 con GC
sns.barplot(df_CG.head(10), x='ClaseReal', y='Aciertos')
plt.title('Top 10 Clases Más Predichas con CG')
plt.xlabel('Clase')
plt.ylabel('Aciertos')
plt.xticks(rotation=45, ha='right')
plt.show()
'''


print("################################################################################################################")
print("################################################################################################################")

# 'Transformación logarítmica (TL)'
df_TL = coincidencias[coincidencias['Algoritmo'] == 'Transformación logarítmica']
df_TL = df_TL.groupby('ClaseReal')['ClasePredPrime'].count().reset_index(name='Aciertos')

# Representa como de importante es cada clase respecto a ese algoritmo
df_TL['Porcentaje'] = ((df_TL['Aciertos'] / df_TL['Aciertos'].sum()) * 100).__round__(3)
df_TL = df_TL.sort_values(by='Aciertos', ascending=False)
aciertos_TL = df_TL['Aciertos'].sum()

# Mostrar el DataFrame resultante
print(df_TL.head(5))
print('Aciertos totales TL', aciertos_TL)


'''
# TOP 10 con TL
sns.barplot(df_TL.head(10), x='ClaseReal', y='Aciertos')
plt.title('Top 10 Clases Más Predichas con TL')
plt.xlabel('Clase')
plt.ylabel('Aciertos')
plt.xticks(rotation=45, ha='right')
plt.show()
'''

print("################################################################################################################")
print("################################################################################################################")

# Algoritmo con más aciertos
col_aciertos = ['Ecualización Histograma', 'Ecualización Histograma Adaptativa Limitada por Contraste', 'Corrección Gamma', 'Transformación Logarítmica']
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
print(mejora_EH.head(10))


'''
# ¿Qué clase mejora más? -> Es decir, que sea más probable de ser predicha
# Se hace la media de todos los elementos que tengan la misma clase
clase_EH = mejora_EH.groupby('ClaseReal')['% Mejora'].mean().__round__(3).reset_index(name='% Mejora Por Clase')
clase_EH = clase_EH.sort_values(by='% Mejora Por Clase', ascending=False)
#print('Top 10 mejores clases')
#print(clase_EH.head(10))


# Grafico 10 clases mas mejoradas con EH
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
print(mejora_EHALC.head(10))


'''
# ¿Qué clase mejora más? -> Es decir, que sea más probable de ser predicha
# Se hace la media de todos los elementos que tengan la misma clase
clase_EHALC = mejora_EHALC.groupby('ClaseReal')['% Mejora'].mean().__round__(3).reset_index(name='% Mejora Por Clase')
clase_EHALC = clase_EHALC.sort_values(by='% Mejora Por Clase', ascending=False)
print('Top 10 mejores clases')
print(clase_EHALC.head(10))

# Grafico 10 clases mas mejoradas con EHALC
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
print(mejora_CG.head(10))


'''
# ¿Qué clase mejora más? -> Es decir, que sea más probable de ser predicha
# Se hace la media de todos los elementos que tengan la misma clase
clase_CG = mejora_CG.groupby('ClaseReal')['% Mejora'].mean().__round__(3).reset_index(name='% Mejora Por Clase')
clase_CG = clase_CG.sort_values(by='% Mejora Por Clase', ascending=False)
print('Top 10 mejores clases')
print(clase_CG.head(10))

# Grafico 10 clases mas mejoradas con CG
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
print(mejora_TL.head(10))


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
ax.pie(niveles['count'], labels=niveles['Nivel'], autopct='%1.1f%%', startangle=0, textprops={'fontsize': 10}, pctdistance=0.8)
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

        '''
        # Grafico que muestre el top por nivel
        sns.barplot(top_clases_nivel[top_clases_nivel['Nivel'] == i], x='ClaseReal', y='% Mejora Por Clase')
        plt.xlabel('Clase Real')
        plt.ylabel('% Mejora Media Por Clase')
        plt.title(f'Top Clases Con {algoritmo} en Nivel {i}')
        # Rotar las etiquetas
        plt.xticks(rotation=45, ha='right')
        # Mostrar el gráfico
        plt.show()
        '''

    niveles_mejora = pd.DataFrame(niveles_mejora, columns=['Nivel', 'Mejora'])

    '''
    # Grafico que compara la mejora por niveles, es decir, los 13 niveles y la mejora en cada uno de ellos
    plt.figure(figsize=(10, 10))
    plt.plot(niveles_mejora['Nivel'], niveles_mejora['Mejora'], marker='o', linestyle='-', color='b')
    plt.title(f'Mejora por Nivel con {algoritmo}')
    plt.xlabel('Nivel')
    plt.ylabel('% Mejora')
    plt.xticks(niveles['Nivel'].values)
    plt.grid(True)
    plt.show()
    '''

    return niveles_mejora

'''
niveles_mejora_EH = comparacion_niveles('EH')
niveles_mejora_EHALC = comparacion_niveles('EHALC')
niveles_mejora_CG = comparacion_niveles('CG')
niveles_mejora_TL = comparacion_niveles('TL')
'''

print("################################################################################################################")
print("################################################################################################################")


def aciertos_jerarquia(algoritmo, nivel):

    df_res = pd.DataFrame

    if algoritmo == 'EH':
        df_res = mejora_EH

    elif algoritmo == 'EHALC':
        df_res = mejora_EHALC

    elif algoritmo == 'TL':
        df_res = mejora_TL

    elif algoritmo == 'CG':
        df_res = mejora_CG

    nv = nivel.split('_')[1]

    # Agrupar por clase
    df_res = df_res.groupby('ClaseReal')[nivel].sum().reset_index()
    df_res = df_res.sort_values(by=nivel, ascending=False)

    '''
    sns.barplot(df_res.head(10), x='ClaseReal', y=nivel)
    plt.xlabel('Clase')
    plt.ylabel('Aciertos')
    plt.title(f'Top 10 Clases Predichas Nivel {nv} Con {algoritmo}')
    # Rotar las etiquetas
    plt.xticks(rotation=45, ha='right')
    # Mostrar el gráfico
    plt.show()
    '''
    return df_res

'''
print("################################################################################################################")
# Ecualización del histograma

aciertosEH1 = aciertos_jerarquia('EH', 'Aciertos_Real')
aciertosEH2 = aciertos_jerarquia('EH', 'Aciertos_Padre')
aciertosEH3 = aciertos_jerarquia('EH', 'Aciertos_Abuelo')

familiaEH = pd.merge(aciertosEH1, aciertosEH2, on='ClaseReal', how='left')
familiaEH = pd.merge(familiaEH, aciertosEH3, on='ClaseReal', how='left')


# FUNCION PARA LA GRÁFICA DE COLUMNAS APILADAS
# Filtrar para mostrar las primeras 10 clases
familiaEH = familiaEH.head(10)

# Crear el gráfico de columnas apiladas
fig, ax = plt.subplots(figsize=(12, 6))

# Configurar posiciones y anchura de las barras
bar_width = 0.5
r = range(len(familiaEH['ClaseReal']))

# Apilar las columnas
p1 = plt.bar(r, familiaEH['Aciertos_Real'], color='b', edgecolor='white', width=bar_width, label='Aciertos_Real')
p2 = plt.bar(r, familiaEH['Aciertos_Padre'], bottom=familiaEH['Aciertos_Real'], color='r', edgecolor='white', width=bar_width, label='Aciertos_Padre')
p3 = plt.bar(r, familiaEH['Aciertos_Abuelo'], bottom=familiaEH['Aciertos_Real'] + familiaEH['Aciertos_Padre'], color='g', edgecolor='white', width=bar_width, label='Aciertos_Abuelo')

# Añadir etiquetas y título
plt.xlabel('Clase')
plt.ylabel('Aciertos')
plt.title('Variabilidad de Aciertos entre Real, Padre y Abuelo con EH')
plt.xticks(r, familiaEH['ClaseReal'], rotation=45, ha='right')
plt.legend()
plt.show()

print("################################################################################################################")
# Ecualización del histograma Adaptativa Limitada por Contraste
aciertosEHALC1 = aciertos_jerarquia('EHALC', 'Aciertos_Real')
aciertosEHALC2 = aciertos_jerarquia('EHALC', 'Aciertos_Padre')
aciertosEHALC3 = aciertos_jerarquia('EHALC', 'Aciertos_Abuelo')

familiaEHALC = pd.merge(aciertosEHALC1, aciertosEHALC2, on='ClaseReal', how='left')
familiaEHALC = pd.merge(familiaEHALC, aciertosEHALC3, on='ClaseReal', how='left')


# FUNCION PARA LA GRÁFICA DE COLUMNAS APILADAS
# Filtrar para mostrar las primeras 10 clases
familiaEHALC = familiaEHALC.head(10)

# Crear el gráfico de columnas apiladas
fig, ax = plt.subplots(figsize=(12, 6))

# Configurar posiciones y anchura de las barras
bar_width = 0.5
r = range(len(familiaEHALC['ClaseReal']))

# Apilar las columnas
p1 = plt.bar(r, familiaEHALC['Aciertos_Real'], color='b', edgecolor='white', width=bar_width, label='Aciertos_Real')
p2 = plt.bar(r, familiaEHALC['Aciertos_Padre'], bottom=familiaEHALC['Aciertos_Real'], color='r', edgecolor='white', width=bar_width, label='Aciertos_Padre')
p3 = plt.bar(r, familiaEHALC['Aciertos_Abuelo'], bottom=familiaEHALC['Aciertos_Real'] + familiaEHALC['Aciertos_Padre'], color='g', edgecolor='white', width=bar_width, label='Aciertos_Abuelo')

# Añadir etiquetas y título
plt.xlabel('Clase')
plt.ylabel('Aciertos')
plt.title('Variabilidad de Aciertos entre Real, Padre y Abuelo con EHALC')
plt.xticks(r, familiaEHALC['ClaseReal'], rotation=45, ha='right')
plt.legend()
plt.show()


print("################################################################################################################")
# Corrección Gamma
aciertosCG1 = aciertos_jerarquia('CG', 'Aciertos_Real')
aciertosCG2 = aciertos_jerarquia('CG', 'Aciertos_Padre')
aciertosCG3 = aciertos_jerarquia('CG', 'Aciertos_Abuelo')

familiaCG = pd.merge(aciertosCG1, aciertosCG2, on='ClaseReal', how='left')
familiaCG = pd.merge(familiaCG, aciertosCG3, on='ClaseReal', how='left')


# FUNCION PARA LA GRÁFICA DE COLUMNAS APILADAS
# Filtrar para mostrar las primeras 10 clases
familiaCG = familiaCG.head(10)

# Crear el gráfico de columnas apiladas
fig, ax = plt.subplots(figsize=(12, 6))

# Configurar posiciones y anchura de las barras
bar_width = 0.5
r = range(len(familiaCG['ClaseReal']))

# Apilar las columnas
p1 = plt.bar(r, familiaCG['Aciertos_Real'], color='b', edgecolor='white', width=bar_width, label='Aciertos_Real')
p2 = plt.bar(r, familiaCG['Aciertos_Padre'], bottom=familiaCG['Aciertos_Real'], color='r', edgecolor='white', width=bar_width, label='Aciertos_Padre')
p3 = plt.bar(r, familiaCG['Aciertos_Abuelo'], bottom=familiaCG['Aciertos_Real'] + familiaCG['Aciertos_Padre'], color='g', edgecolor='white', width=bar_width, label='Aciertos_Abuelo')

# Añadir etiquetas y título
plt.xlabel('Clase')
plt.ylabel('Aciertos')
plt.title('Variabilidad de Aciertos entre Real, Padre y Abuelo con CG')
plt.xticks(r, familiaCG['ClaseReal'], rotation=45, ha='right')
plt.legend()
plt.show()

print("################################################################################################################")
# Transformación Logarítmica
aciertosTL1 = aciertos_jerarquia('TL', 'Aciertos_Real')
aciertosTL2 = aciertos_jerarquia('TL', 'Aciertos_Padre')
aciertosTL3 = aciertos_jerarquia('TL', 'Aciertos_Abuelo')

familiaTL = pd.merge(aciertosTL1, aciertosTL2, on='ClaseReal', how='left')
familiaTL = pd.merge(familiaTL, aciertosTL3, on='ClaseReal', how='left')


# FUNCION PARA LA GRÁFICA DE COLUMNAS APILADAS
# Filtrar para mostrar las primeras 10 clases
familiaTL = familiaTL.head(10)

# Crear el gráfico de columnas apiladas
fig, ax = plt.subplots(figsize=(12, 6))

# Configurar posiciones y anchura de las barras
bar_width = 0.5
r = range(len(familiaTL['ClaseReal']))

# Apilar las columnas
p1 = plt.bar(r, familiaTL['Aciertos_Real'], color='b', edgecolor='white', width=bar_width, label='Aciertos_Real')
p2 = plt.bar(r, familiaTL['Aciertos_Padre'], bottom=familiaTL['Aciertos_Real'], color='r', edgecolor='white', width=bar_width, label='Aciertos_Padre')
p3 = plt.bar(r, familiaTL['Aciertos_Abuelo'], bottom=familiaTL['Aciertos_Real'] + familiaTL['Aciertos_Padre'], color='g', edgecolor='white', width=bar_width, label='Aciertos_Abuelo')

# Añadir etiquetas y título
plt.xlabel('Clase')
plt.ylabel('Aciertos')
plt.title('Variabilidad de Aciertos entre Real, Padre y Abuelo con TL')
plt.xticks(r, familiaTL['ClaseReal'], rotation=45, ha='right')
plt.legend()
plt.show()



print("################################################################################################################")

# Aciertos para ClaseReal
print(aciertosEH1)
print(aciertosEHALC1)
print(aciertosCG1)
print(aciertosTL1)

print("################################################################################################################")

# Aciertos para ClasePadre
print(aciertosEH2)
print(aciertosEHALC2)
print(aciertosCG2)
print(aciertosTL2)


print("################################################################################################################")

# Aciertos para ClaseAbuelo
print(aciertosEH3)
print(aciertosEHALC3)
print(aciertosCG3)
print(aciertosTL3)
'''

print("################################################################################################################")
# Grafico de aciertos para General
aciertosEH4 = aciertos_jerarquia('EH', 'Aciertos_Totales')
aciertosEHALC4 = aciertos_jerarquia('EHALC', 'Aciertos_Totales')
aciertosCG4 = aciertos_jerarquia('CG', 'Aciertos_Totales')
aciertosTL4 = aciertos_jerarquia('TL', 'Aciertos_Totales')

print(aciertosEH4)
print(aciertosEHALC4)
print(aciertosCG4)
print(aciertosTL4)

'''
sns.barplot(aciertosEH4.head(10), x='ClaseReal', y='Aciertos_Totales')
plt.xlabel('Clase')
plt.ylabel('Aciertos')
plt.title('Top 10 Clases Predichas Nivel General Con EH')
# Rotar las etiquetas
plt.xticks(rotation=45, ha='right')
# Mostrar el gráfico
plt.show()

sns.barplot(aciertosEHALC4.head(10), x='ClaseReal', y='Aciertos_Totales')
plt.xlabel('Clase')
plt.ylabel('Aciertos')
plt.title('Top 10 Clases Predichas Nivel General Con EHALC')
# Rotar las etiquetas
plt.xticks(rotation=45, ha='right')
# Mostrar el gráfico
plt.show()

sns.barplot(aciertosCG4.head(10), x='ClaseReal', y='Aciertos_Totales')
plt.xlabel('Clase')
plt.ylabel('Aciertos')
plt.title('Top 10 Clases Predichas Nivel General Con CG')
# Rotar las etiquetas
plt.xticks(rotation=45, ha='right')
# Mostrar el gráfico
plt.show()

sns.barplot(aciertosTL4.head(10), x='ClaseReal', y='Aciertos_Totales')
plt.xlabel('Clase')
plt.ylabel('Aciertos')
plt.title('Top 10 Clases Predichas Nivel General Con TL')
# Rotar las etiquetas
plt.xticks(rotation=45, ha='right')
# Mostrar el gráfico
plt.show()
'''