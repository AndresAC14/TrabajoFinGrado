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

# print(df_filtered)

# Filtrar las filas donde ClaseReal y ClasePred son iguales
df_filtered = df[df['ClaseReal'] == df['ClasePred']]

# Agrupar las que son iguales y contarlas
df_1 = df_filtered.groupby('ClaseReal')['ClasePred'].count().reset_index(name='Aciertos')
df_2 = df.groupby('ClaseReal')['ClasePred'].count().reset_index(name='Total')

df_Res = pd.merge(df_1, df_2, on='ClaseReal', how='left')

# Crea la columna resultado
df_Res['Porcentaje'] = ((df_Res['Aciertos'] / df_Res['Total']) * 100).__round__(2)

print(df_Res)

# Ahora hacer una grafica que ponga la ClaseReal respecto al Porcentaje
# sns.barplot(df_Res, x=df_Res['ClaseReal'], y=df_Res['Porcentaje'])
# plt.show()

# Eso es demasiado grande, asi que mejor hacer la grafica con los que tienen mas porcentaje
# mayores = df_Res[df_Res['Porcentaje'] > 70]
# sns.barplot(mayores, x='ClaseReal', y='Porcentaje')
# plt.show()


# ClasePred vs ClasePredPrime
# Contar cuantos cambian y que algoritmo estan usando

print("################################################################################################################")
# Algoritmos que manejamos:  ['Ecualización del histograma'
#  'Ecualización del histograma adaptativa limitada por contraste'
#  'Corrección de gamma' 'Transformación logarítmica']
'''
# Aciertos para el algoritmo Ecualizacion del histograma
df_ecualizacion1 = df[(df['ClaseReal'] == df['ClasePred']) & (df['Algoritmo'] == 'Ecualización del histograma')]

df_ecualizacion1 = df_ecualizacion1.groupby('ClaseReal')['ClasePred'].count().reset_index(name='Aciertos')

# Mostrar el DataFrame resultante
print(df_ecualizacion1.head(5))
print(df_ecualizacion1['Aciertos'].sum())


def aciertosAlgoritmos(df):
    algoritmos = df['Algoritmo'].unique()
    df_resultado = pd.DataFrame()

    for alg in algoritmos:
        # Filtrar las filas donde claseReal es igual a clasePred y Algoritmo es igual al valor actual
        df_algoritmo_aciertos = df[(df['ClaseReal'] == df['ClasePred']) & (df['Algoritmo'] == alg)]

        # Contar los aciertos para cada clase y reiniciar el índice
        df_aciertos = df_algoritmo_aciertos.groupby('ClaseReal')['ClasePred'].count().reset_index(name='Aciertos '+alg)

        # Concatenar el DataFrame resultante al DataFrame general
        df_resultado = pd.concat([df_resultado, df_aciertos], axis=1)

    return df_resultado
'''

def aciertosAlgoritmos(df, algoritmos):
    df_resultado = pd.DataFrame()

    for alg in algoritmos:
        # Filtrar las filas donde claseReal es igual a clasePred y Algoritmo es igual al valor actual
        df_algoritmo_aciertos = df[(df['ClaseReal'] == df['ClasePred']) & (df['Algoritmo'] == alg)]

        # Contar los aciertos para cada clase y reiniciar el índice
        df_aciertos = df_algoritmo_aciertos.groupby('ClaseReal').size().reset_index(name='Aciertos '+alg)

        # Concatenar el DataFrame resultante al DataFrame general
        if df_resultado.empty:
            df_resultado = df_aciertos
        else:
            df_resultado = pd.concat([df_resultado, df_aciertos['Aciertos '+alg]])

    return df_resultado


# No se por que al poner el codigo este dentro de la otra funcion no funciona, es raro
def aciertos(df):
    algoritmos = df['Algoritmo'].unique()
    df_aciertos = aciertosAlgoritmos(df, algoritmos)
    df_aciertos = df_aciertos.replace(np.nan, 0)

    for alg in algoritmos:
            df_aciertos['Aciertos ' + alg] = df_aciertos['Aciertos ' + alg].astype(int)

    return df_aciertos


df_aciertos = aciertos(df)
print(df_aciertos.head(10))
print("###############################################################################################################")
# Ahora quiero sumar los valores de cada columna y ponerlos en una grafica, es decir barplot con los algoritmos
 # NO FUNCIONA
def suma(df, aciertos):
    algoritmos = df['Algoritmo'].unique()
    suma = pd.DataFrame()

    for alg in algoritmos:
        suma[alg] = aciertos['Aciertos ' + alg].sum()

    return suma

gr = suma(df, df_aciertos)
print(gr.head(5))
