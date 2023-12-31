# -*- coding: utf-8 -*-
"""Market_Value.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ffy3pAV0REee2w3-mbuJvMlm3bpgN4hB
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel('/content/Final.xlsx')

data

data = data.dropna()

data.isnull().sum()

data.columns

"""Se procede a tomar un dataset mas pequeño para un mejor analisis y predicción

"""

data.drop(columns = ['Unnamed: 0.1','Unnamed: 0','Player','Nation','Club_x','MP','Starts','PK_x','PKatt_x','CrdY','CrdR','FK','Sh/90', 'SoT/90', 'G/Sh', 'G/SoT','TakleD', 'Succ_x', '%','ShotB', 'PassB', 'Int', 'Clr', 'Cmp%', 'Touches', 'Succ_y', 'Att', 'Succ%', '#Pl','PK','PKatt','Sh'], inplace=True)
data

data.iloc[141]

data.sample(5)

"""Recién vi irregularidades en la columna **Market Value**
Procedo a hacer la respectiva limpieza
"""

data['Market value'] = data['Market value'].apply(lambda x: x.replace("€",""))
data

data['Market value'] = data['Market value'].apply(lambda x: x.replace("m",""))
data

data['Market value'] = data['Market value'].astype('float64')

def graficos_eda_categoricos(cat):

    #Calculamos el número de filas que necesitamos
    from math import ceil
    filas = ceil(cat.shape[1] / 2)

    #Definimos el gráfico
    f, ax = plt.subplots(nrows = filas, ncols = 2, figsize = (16, filas * 6))

    #Aplanamos para iterar por el gráfico como si fuera de 1 dimensión en lugar de 2
    ax = ax.flat

    #Creamos el bucle que va añadiendo gráficos
    for cada, variable in enumerate(cat):
        cat[variable].value_counts().plot.barh(ax = ax[cada])
        ax[cada].set_title(variable, fontsize = 12, fontweight = "bold")
        ax[cada].tick_params(labelsize = 12)

graficos_eda_categoricos(data.select_dtypes('O'))

def estadisticos_cont(num):
    #Calculamos describe
    estadisticos = num.describe().T
    #Añadimos la mediana
    estadisticos['median'] = num.median()
    #Reordenamos para que la mediana esté al lado de la media
    estadisticos = estadisticos.iloc[:,[0,1,8,2,3,4,5,6,7]]
    #Lo devolvemos
    return(estadisticos)

estadisticos_cont(data.select_dtypes('number'))

from sklearn.preprocessing import OneHotEncoder

#Categóricas
cat = data.select_dtypes('O')

#Instanciamos
ohe = OneHotEncoder(sparse = False)

#Entrenamos
ohe.fit(cat)

#Aplicamos
cat_ohe = ohe.transform(cat)

#Ponemos los nombres
cat_ohe = pd.DataFrame(cat_ohe, columns = ohe.get_feature_names_out(input_features = cat.columns)).reset_index(drop = True)

cat_ohe

num = data.select_dtypes('number').reset_index(drop = True)

df_ml = pd.concat([num,cat_ohe], axis = 1)
df_ml

from random import sample

print(df_ml.columns[1:], "\n")
print(sample(set(df_ml.columns[1:]), 3))

data['Market value'] = data['Market value'].astype('int64')

x = data.drop(columns=['Market value','Leauge','Pos','Gls90', 'Ast90', 'G+A','SoT',
       'SoT%', 'TackleW', 'Tkl%'])
y = data['Market value']

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.3)

from sklearn.ensemble import RandomForestClassifier

bosque = RandomForestClassifier(n_estimators=100,
                               criterion="gini",
                               max_features="sqrt",
                               bootstrap=True,
                               max_samples=2/3,
                               oob_score=True)

bosque.fit(x, data["Market value"].values)

#print(bosque.predict([[50, 16, 1, 1, 40]]))
print(bosque.score(x, data["Market value"].values))
print(bosque.oob_score_)

x.columns

# Predicción
print(bosque.predict([[27,5000,5,10,15,145,1200,88,2680,3600]]))

#Datos de jugador random
print('Valor de mercado',data['Market value'].iloc[70],'\n',x.iloc[70])