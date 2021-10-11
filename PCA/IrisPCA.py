from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

# PCA
def PCA_DATA(x_true):

    x_true -= np.mean(x_true, axis=0)  # 去均值化

    cov = np.dot(x_true.T, x_true) / x_true.shape[0]  # 求协方差矩阵

    # 对协方差矩阵进行SVD分解
    U, S, V = np.linalg.svd(cov)
    x_true_rot = np.dot( x_true, U[0:2].T)

    return x_true_rot

def picture(trans_data):
    index1 = np.where(iris.target == 0)
    index2 = np.where(iris.target == 1)
    index3 = np.where(iris.target == 2)

    labels = ['setosa', 'versicolor', 'virginica']

    plt.plot(trans_data[index1][:, 0], trans_data[index1][:, 1], 'r*')
    plt.plot(trans_data[index2][:, 0], trans_data[index2][:, 1], 'g*')
    plt.plot(trans_data[index3][:, 0], trans_data[index3][:, 1], 'b*')
    plt.legend(labels)

# 载入鸢尾花数据集
iris = load_iris()
x_true = iris.data

# 利用自定义PCA模块处理数据
x_true_rot = PCA_DATA(x_true)

# 调用库函数PCA
pca = PCA(n_components=2)                           #设置保留的主成分个数为2
trans_data = pca.fit_transform(iris.data)           #调用fit_transform方法，返回新的数据集

# 原始数据的前两维features
plt.figure(figsize=(10, 10))
plt.subplot(2,2,1)
plt.title("Before PCA data's first two features")
picture(x_true)

# 利用库函数PCA处理后
plt.subplot(2,2,3)
plt.title("After package PCA data's first two features")
picture(trans_data)

# PCA处理后数据的前两维features
plt.subplot(2,2,4)
plt.title("After custom PCA data's first two features")
picture(x_true_rot)
plt.show()

# 利用PCA后的前两维features对数据进行SVM分类
clf = svm.SVC()
clf.fit(x_true_rot[:, 0:2], iris.target.reshape(iris.target.shape[0], ))
y_predict = clf.predict(x_true_rot[:, 0:2])

# 评价分类准确率
score = clf.score(x_true_rot[:, 0:2], iris.target)
print("利用自定义PCA后的前两维度特征对鸢尾花数据集分类精度为：")
print("score= ", score)





