import numpy as np #импортируем библиотеку

y = np.loadtxt(fname='data.csv', delimiter=';', encoding='utf-8') #открываем файл с данными
y_max, y_min = np.max(y), np.min(y) #находим максимум и минимум
y_max_ix = np.where(y == y_max) #находим индексы минимума
y_min_ix = np.where(y == y_min) # и максимума

y[y_max_ix[0][0], y_max_ix[1][0]],  y[y_min_ix[0][0], y_min_ix[1][0]] = y_min, y_max #меняем местами