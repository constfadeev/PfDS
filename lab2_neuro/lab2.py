'''Данный скрипт угадывает четное или нечетное число'''

import numpy as np
import glob
from PIL import Image

'''переводит картинку в одномерный массив'''
def image_to_array(path): 
    img = Image.open(path)
    arr = np.array(img)
    a = np.zeros(0)
    for i in arr:
        a = np.concatenate((a, np.transpose(i)[-1]), axis=None)
    return np.where(a==255., 1, a)


'''создает тренировочный датасет с выходом матрицы значений X и соответсвующего значения y'''
def train_ds(DIR_1):
    files = glob.glob(f"{DIR_1}\*.png")
    all_png = np.zeros(25)
    val = np.zeros(1)
    for file in files:
        all_png = np.vstack((all_png, image_to_array(file)))
        val = np.vstack((val, int(file[-5])))
    #print(all_png[1:], val[1:])
    return all_png[1:], val[1:]


DIR = r'.\train' #расположение файлов для обучения

D, Y0 = train_ds(DIR) #обучаем на файлах для обучения

w = np.zeros(25)

α = 0.2
β = -0.4
σ = lambda x: 1 if x > 0 else 0 #Коэффциент, зависящий от x


def f(x):
    s = β + np.sum(x @ w) #входной вектор на весовой коэффициент
    return σ(s)
 
def train():
    global w #берем веса
    _w = w.copy() #копируем для запоминания
    for x, y in zip(D, Y0): #итерируется по x и y в D и Y0, грубо говоря идет сопоставление x в D, а y в Y0
        w += α * (y - f(x)) * x #рассчитываются новые веса
        print(w)
    return (w != _w).any()
           
while train(): # работает пока w и _w не совпадет
    print(w)
    
print('Вывод:') 
for i in glob.glob(r".\test\*.png"): #берем каждый файл в папке тест
    print(f'"{i[-7]}" - {f(image_to_array(i))}') #применяем на него функцию поиска коэффициентов, 0 - нечетное, 1 - четное
