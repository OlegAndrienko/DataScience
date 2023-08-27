import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error




#напишіть функцію гіпотези лінійної регресії у векторному вигляді; h(x) = W * X
# X - матриця признаків
# W - вектор параметрів
# y - вектор значень цільової змінної
def h(X,W):            #функція повертає вектор значень цільової змінної
    return np.dot(X, W) #np.dot(X, W) - множення матриць

#функція для обчислення коєфіцієнтів лінійної регресії
# X - матриця признаків
# y - вектор значень цільової змінної
def linear_regression(X, y):   #функція повертає вектор параметрів W = (X^T * X)^-1 * X^T * y
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

#створіть функцію для обчислення функції втрат у векторному вигляді  за допомогою LinearRegression
# X - матриця признаків
# W - вектор параметрів
# y - вектор значень цільової змінноїі 

def loss_sklearn(X, y):
    model = LinearRegression().fit(X, y)
    return mean_squared_error(y, model.predict(X))

#створіть функцію для обчислення функції втрат у векторному вигляді
# X - матриця признаків
# W - вектор параметрів
# y - вектор значень цільової змінноїі 
def loss(X, y, W):
    return np.sum((h(X, W) - y) ** 2) / len(X)

def J(X, y, W):
    return np.sum((h(X,W) - y)**2) / (2 * len(y)) 


#реалізуйте один крок градієнтного спуску у векторному вигляді
# X - матриця признаків
# W - вектор параметрів
# y - вектор значень цільової змінної
# alpha - швидкість навчання

def gradient_step(X, y, W, alpha):
    return W - alpha * 2 / len(X) * np.dot(X.T, (h(X,W) - y))

#реалізуйте функцію навчання лінійної регресії градієнтним спуском
# X - матриця признаків
# y - вектор значень цільової змінної
# alpha - швидкість навчання
# num_iter - кількість ітерацій

def gradient_descent(X, y, alpha, num_iter): #функція повертає вектор параметрів W
    W = np.zeros((X.shape[1], 1))
    for i in range(num_iter):
        W = gradient_step(X, y, W, alpha)
    return W

print("------------------------csv---------------------------------------")
houses = pd.read_csv('Housing.csv')
print(houses.head())

#знайдіть найкращі параметри $\vec{w}$ для датасету прогнозуючу ціну на будинок залежно від
# площі, кількості ванних кімнат та кількості спалень;
print("---------------------------new_houses------------------------------------")

price = np.array(houses['price'])
print(f"price = {price}")
print(f"shape price = {price.shape}")

price_array = []
for i in range(len(price)):
    price_array.append([price[i]])
    
print(f"price = {price_array}")
print(f"shape price = {np.array(price_array).shape}")
print("---------------------------new_houses------------------------------------")
area = np.array(houses['area'])
print(f"area = {area}")
bedrooms = np.array(houses['bedrooms'])
print(f"bedrooms = {bedrooms}")
bathrooms = np.array(houses['bathrooms'])
print(f"bathrooms = {bathrooms}")

f0 = []
for i in range(len(area)):
    f0.append(1)
print(f"f0 = {f0}")
F_0 = np.array(f0)
print(f"F0 = {F_0}")
print(f"shape F0 = {F_0.shape}")

print("---------------------------X------------------------------------")

X = np.array([F_0, area, bedrooms, bathrooms]).T
print(f"X = {X}")

#розрахуємо коєфіцієнти
W = linear_regression(X, price_array)
print(f"W = {W}")

#Перевіряємо розрахунок за домоиогою бібліотеки sklearn
model = LinearRegression().fit(X, price_array)
W_sklearn = model.coef_.T
W_sklearn[0] = model.intercept_
print(f"W_sklearn = {W_sklearn.T[0]}")

#коефіцієнти різні тому що в sklearn використовується інша функція втрат ???

#знайдіть найкращі параметри $\vec{w}$ для датасету прогнозуючу ціну на будинок 
# залежно від площі та кількості ванних кімнат та кількості спалень;


res = gradient_descent(X, price_array, 0.0000001, 1000)
print(f"res = {res}")



