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
    val = np.zeros(3)
    for file in files:
        all_png = np.vstack((all_png, image_to_array(file)))
        val = np.vstack((val, [int(x) for x in list(file[-7:-4])]))
    print(all_png[1:], val[1:])
    return all_png[1:], val[1:]

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
    print(w)

for i in range(8):
    print(f(D[i]))