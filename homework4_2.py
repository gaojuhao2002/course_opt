#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:GJH 
# time:2022/10/17
import math
import numpy as np
import matplotlib.pyplot as plt

class p92_func():
    def __init__(self, x=None):
        if x is None:
            print("请输入x")
        else:
            self.x = np.asarray(x).flatten()

    def f(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        x1 = x[0]
        x2 = x[1]
        return 0.5 * x1 * x1 * ((x1 * x1 / 6) + 1) + x2 * math.atan(x2) - 0.5 * math.log((x2 * x2 + 1))

    def g(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        x1 = x[0]
        x2 = x[1]
        g = np.zeros(2, )
        g[0] = (x1 ** 3) / 3 + x1
        g[1] = math.atan(x2)
        return g

    def G(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        x1 = x[0]
        x2 = x[1]
        G = np.zeros((2, 2))
        G[0, 0] = x1 * x1 + 1
        G[1, 1] = 1 / (1 + x2 * x2)
        return G

def BasicNewton(getf, getg, getG, x, alpha=1, eps=1e-5, iter=10, x_k=[], f_k=[]):
    #第一次先加进来
    x = np.asarray(x)
    x_k.append(x)

    for k in range(iter):
        g = getg(x)     #计算当前点的梯度

        f = getf(x)  # 计算当前点的函数值
        f_k.append(f) # 添加这一轮的函数值
        # 满足终止条件直接结束
        if np.linalg.norm(g) < eps:
            print('查找第{}次发现,函数最优解为{}，函数的最优值为{},'
                  '此时梯度的范数为{}'.format(k, x, f, np.linalg.norm(g)))
            break
        # 不满足终止条件
        try:
            d = -np.dot(np.linalg.inv(getG(x)), g.T)  # 计算牛顿方向
        except:
            print('第{}次，此时的Hessian矩阵不可逆'.format(k))
        x = x + alpha * d  # 更新迭代点
        # 超过迭代次数也没找到
        if k == iter - 1:
            print("经过了{}次搜索没有找到满足终止条件的最优值".format(iter))
            break
        x_k.append(x)  # 添加这一轮的自变量
    #循环外返回结果数组
    x_k=np.asarray(x_k)
    f_k=np.asarray(f_k)
    return x_k, f_k

if __name__ == '__main__':
    print("92页实验1")
    # 题目给的两个初始点
    x0_1 = np.array([1, 0.7])
    x0_2 = np.array([1, 2])

    # 分别创建对象
    sub1 = p92_func(x0_1)
    sub2 = p92_func(x0_2)

    # 分别调用基本牛顿法，并保留迭代点和函数值
    print('初始值为[1,0.7]')
    x_k1, f_k1 = BasicNewton(sub1.f, sub1.g, sub1.G, x0_1,x_k=[],f_k=[])
    print('初始值为[1,2]')
    #如果x_k=[]不写，这次调用会继承上一次的
    x_k2, f_k2 = BasicNewton(sub2.f, sub2.g, sub2.G, x0_2,x_k=[],f_k=[])

    plt.figure(1)
    plt.plot(x_k1[:,0],x_k1[: ,1],c='b')
    plt.scatter(np.arange(len(f_k1)),f_k1,c='r')
    plt.legend(['x_k','f_k'])
    plt.title('x_0=[1,0.7]')
    plt.xlabel('x1')
    plt.ylabel('x2/f_k')
    plt.show()

    plt.figure(2)
    plt.plot(x_k2[:,0],x_k2[: ,1],c='b',linestyle='--')
    plt.scatter(np.arange(len(f_k2)), f_k2, c='r',marker='*')
    plt.legend(['x_k','f_k'])
    plt.title('x_0=[1,2]')
    plt.xlabel('x1')
    plt.ylabel('x2/f_k')
    plt.show()

