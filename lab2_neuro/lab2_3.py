'''Перцептрон, обучаясь на тренировочных данных, вычисляет какое число изображено на картинке в восьмеричной системе счисления при повороте картинки на угол от 0 до 90 градусов''' 

import numpy as np
from scipy import ndimage
import glob
from PIL import Image

'''переводит картинку в одномерный массив''' 
def image_to_array(path, rotate=0): 
    img = Image.open(path)
    img_rot = ndimage.rotate(img, rotate, reshape=False)
    arr = np.array(img_rot)
    a = np.zeros(0)
    for i in arr:
        a = np.concatenate((a, np.transpose(i)[-1]), axis=None)
    return np.where(a>=128., 1, 0)

'''создает тренировочный датасет с выходом матрицы значений X и соответсвующего значения y'''
def train_ds(DIR_1):
    files = glob.glob(f"{DIR_1}\*.png")
    all_png = np.zeros(25)
    val = np.zeros(3)
    for file in files:
        all_png = np.vstack((all_png, image_to_array(file)))
        val = np.vstack((val, [int(x) for x in list(file[-7:-4])]))
    #print(all_png[1:], val[1:])
    return all_png[1:], val[1:]

def test_to_rotate(DIR_2, rotate=0):
    files = glob.glob(f"{DIR_2}\*.png")
    all_png = np.zeros(25)
    for file in files:
        all_png = np.vstack((all_png, image_to_array(file, rotate)))
    return all_png[1:]

DIR = r'.\train_8' #расположение файлов для обучения

D, Y0 = train_ds(DIR) #обучаем на файлах для обучения

w = np.zeros((D[0].shape[0],Y0[0].shape[0]))

β = -0.4
 
α = 0.2
 
σ = lambda x: (x > 1).astype(int)

def f(x):
    s = β + x @ w
    return σ(s)
 
def train():
    global w
    _w = w.copy()
    for x, y in zip(D, Y0):
        i = np.where(x>0)
        w[i] += α * (y - f(x))
    return (w != _w).any()
           
while train():
    #print(w)
    pass

#Вращение картинки
eq_sum = 0
for i in range(0,100,10):
    eq = 0
    D_rot = test_to_rotate(DIR, rotate=i)
    for j in range(8):
        if np.array_equal(f(D[j]), f(D_rot[j])): #если получившиеся значения равны, плюсуем
            eq += 1
    print(f'При повороте на {i} градусов привильно угадано {eq} чисел, {eq/8*100}%')
    eq_sum += eq

print(f'Всего правильно угаданных значений при повороте картинки: {eq_sum}')
print(f'Процент правильно угаданных значений: {eq_sum/10/8*100}')

'''
При повороте на 0 градусов привильно угадано 8 чисел, 100.0%
При повороте на 10 градусов привильно угадано 7 чисел, 87.5%
При повороте на 20 градусов привильно угадано 5 чисел, 62.5%
При повороте на 30 градусов привильно угадано 3 чисел, 37.5%
При повороте на 40 градусов привильно угадано 0 чисел, 0.0%
При повороте на 50 градусов привильно угадано 1 чисел, 12.5%
При повороте на 60 градусов привильно угадано 1 чисел, 12.5%
При повороте на 70 градусов привильно угадано 1 чисел, 12.5%
При повороте на 80 градусов привильно угадано 1 чисел, 12.5%
При повороте на 90 градусов привильно угадано 1 чисел, 12.5%
Всего правильно угаданных значений при повороте картинки: 28
Процент правильно угаданных значений: 35.0
'''
