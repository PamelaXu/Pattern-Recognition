import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


def imageToGif(inputName, outfileName):
    files = os.listdir(inputName)
    print(files)
    frames = []
    for file in files:
        frames.append(imageio.imread(inputName + '\\' + file))
    imageio.mimsave(outfileName, frames, 'GIF', duration=0.01)


def make_point(point_number, dim, scale):  # 生成分布于sum(xi)-scale=0两旁的点
    """
    生成分类点
    :param point_number: 点的数目（int)
    :param dim: 点的维数(int)
    :param scale: 点的范围(int)
    :return:
    """
    # np.random.seed(10)
    X = np.random.random([point_number, dim]) * scale
    Y = np.zeros(point_number)
    sum_X = np.sum(X, axis=1)
    for index in range(point_number):
        if sum_X[index] - scale < 0:
            Y[index] = -1
        else:
            Y[index] = 1
    return X, Y


class Plotting(object):  #  画图函数
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def open_in(self):
        plt.ion()

    def close(self):
        plt.ioff()
        plt.show()

    def vis_plot(self, weight, b, number):
        plt.cla()
        plt.xlim(0, np.max(self.X.T[0]) + 1)
        plt.ylim(0, np.max(self.X.T[1]) + 1)
        plt.scatter(self.X.T[0], self.X.T[1], c=self.Y)
        if True in list(weight == 0):
            plt.plot(0, 0)
        else:
            x1 = -b / weight[0]
            x2 = -b / weight[1]
            plt.plot([x1, 0], [0, x2])
        plt.title('change time:%d' % number)
        number1 = "%05d"%number
        if number > 450:
            plt.savefig(r'pil\%s.png' % number1)
        plt.pause(0.01)

    def just_plot_result(self, weight, b):
        plt.scatter(self.X.T[0], self.X.T[1], c=self.Y)
        x1 = -b / weight[0]
        x2 = -b / weight[1]
        plt.plot([x1, 0], [0, x2])
        plt.show()

class PerceptionMethod(object):  # 定义 感知机学习 类
    def __init__(self, X, Y, eta):  # 类中参数是 X,Y（X,Y)均为numpy数组,eta,eta是学习率
        if X.shape[0] != Y.shape[0]:  # 要求X,Y中的数目一样，即一个x对应一个y,否则返回错误
                raise ValueError('Error,X and Y must be same when axis=0 ')
        else:  # 在类中储存参数
                self.X = X
                self.Y = Y
                self.eta = eta

    def ini_Per(self):  # 感知机的原始形式
        weight = np.zeros(self.X.shape[1])  # 初始化weight,b
        b = 0
        number = 0  # 记录训练次数
        mistake = True  # mistake是变量用来说明分类是否有错误
        while mistake is True:  # 当有错时
            mistake = False  # 开始下一轮纠错前需要将mistake变为true，一来判断这一轮是否有错误
            for index in range(self.X.shape[0]):  # 循环开始
                if self.Y[index] * (weight @ self.X[index] + b) <= 0:  # 错误判断条件
                    weight += self.eta * self.Y[index] * self.X[index]  # 进行更新weight，b
                    b += self.eta * self.Y[index]
                    number += 1
                    print(weight, b)
                    mistake = True  # 此轮检查出错误，表明mistake为true，进行下列一轮
                    break  # 找出第一个错误后调出循环
        return weight, b  # 返回值

    def plot_ini_Per(self):
        if self.X.shape[1] != 2:
            raise ValueError("dimension doesn't support")
        else:
            weight = np.zeros(self.X.shape[1])
            b = 0
            number = 0
            mistake = True
            Vis = Plotting(self.X, self.Y)
            while mistake is True:
                mistake = False
                Vis.open_in()
                Vis.vis_plot(weight, b, number)
                for index in range(self.X.shape[0]):
                    if self.Y[index] * (weight @ self.X[index] + b) <= 0:
                        weight += self.eta * self.Y[index] * self.X[index]
                        b += self.eta * self.Y[index]
                        number += 1
                        print('error time:', number)
                        print(weight, b)
                        mistake = True
                        break
            Vis.close()
        return weight, b

    def dual_Per(self):  #  感知机的对偶形式
        Gram = np.dot(self.X, self.X.T)
        alpha = np.zeros(self.X.shape[0])
        b = 0
        mistake = True
        while mistake is True:
            mistake = False
            for index in range(self.X.shape[0]):
                if self.Y[index] * (alpha * self.Y @ Gram[index] + b) <= 0:
                    alpha[index] += self.eta
                    b += self.eta * self.Y[index]
                    print(alpha, b)
                    mistake = True
                    break
        weight = self.Y * alpha @ self.X
        return weight, b


if __name__ == '__main__':
    # imageToGif(r'D:\py文件\zhihu_code\machine_learning\pil', 'my1.GIF')
    X = np.array([[3, 3], [4, 3], [1, 1]])
    Y = np.array([1, 1, -1])
    ##############################################################
    # X, Y = make_point(15, 2, 10)
    PER = PerceptionMethod(X, Y, 1)
    # weight, b = PER.plot_ini_Per()
    print(PER.ini_Per())
    print(PER.dual_Per())
    #############################################################
    # vis = Plotting(X, Y)
    # vis.just_plot_result(weight, b)
    # dual_perception(X, Y, 1)