'''多变量线性回归'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = U'D:/1机器学习/线性回归/ex1/ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
print(data2.head())

'''特征归一化'''
data2 = (data2 - data2.mean()) / data2.std()
print(data2.head())

'''计算代价函数'''
def computeCost(X, y, theta):
    inner =np.power((X * theta.T) - y, 2)
    return np.sum(inner) / (2 * len(X))

'''梯度下降,更新theta'''
def gradientDesecent(X, y, theta, alpha, iters):  # alpha：学习率 iters：迭代次数
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1]) #ravel()将数组拉成一维数组
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

# 加一常数项
data2.insert(0,'Ones', 1)

cols = data2.shape[1]
X2 = data2.iloc[:,0:-1]
y2 = data2.iloc[:,cols-1:cols]
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta = np.matrix(np.array([0,0,0]))

g, cost = gradientDesecent(X2, y2, theta, alpha=0.01, iters=1500)
print(g)

'''正规方程'''
def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y  # X.T@X等价于X.T.dot(x)
    return theta

final_theta = normalEqn(X2, y2)
print(final_theta)