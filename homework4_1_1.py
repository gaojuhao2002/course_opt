#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:GJH 
# time:2022/10/17
# 采用精确线性搜索的最速下降法求解课表例题3.1
import numpy as np
# 一般形式f=0.5*xGx+bx+c
class Square_function():
    def __init__(self, x=None, G=None, b=None, c=0):
        if x is None:
            print("请输入x的值")
        else:
            self.x = np.asarray(x).flatten()
        if G is None:
            print("请输入Hessian矩阵")
        else:
            self.G = np.asarray(G)
        if b is None:
            print('请输入一次项系数b')
        else:
            self.b = np.asarray(b).flatten()
        self.c=c
    def f(self,x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        G = self.G
        b = self.b
        c = self.c
        return 0.5 * np.dot(x, np.dot(G, x.T)) + np.dot(b, x.T) + c

    def g(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        G = self.G
        b = self.b
        # 返回Gx+b
        return np.dot(G, x.T) + b.T


# 精确最速下降的函数
def acmaxdown(getf, getg, G, x, epsilon=1e-5):
    x = np.asarray(x)
    G = np.asarray(G)
    iter = 1000

    for k in range(iter):
        g = getg(x)  # 计算梯度此时g为行向量

        # 如果终止条件满足直接结束(满足g的范数低于给定值)
        if np.linalg.norm(g) < epsilon:
            print('当G={}时\n查找第{}次发现,函数最优解为{}，函数的最优值为{},'
                  '此时梯度的范数为{}'.format(G, k,x, getf(x),np.linalg.norm(g)))
            break

        # 若不满足最优条件
        d = -g                                      # 计算负梯度方向
        alpha = np.dot(g, g.T) / np.dot(g, np.dot(G,g.T))  # 计算步长
        x = x + alpha * d                           # 更新x的值重新进入循环
        if k == iter-1:
            print("经过了{}次搜索没有找到满足终止条件的最优值".format(iter))
            break


if __name__ == '__main__':
    # 例题3.1
    print('例题3.1求解')
    # 初始化参数
    b = np.array([2, 3])
    c = 10
    x0 = np.array([-30, 100])
    G1 = np.array([[21, 4], [4, 15]])
    G2 = np.array([[21, 4], [4, 1]])

    # 做题创建对象
    sub1 = Square_function(x0, G1, b, c)
    sub2 = Square_function(x0, G2, b, c)

    #搜索
    acmaxdown(sub1.f, sub1.g, G1, x0)
    acmaxdown(sub2.f, sub2.g, G2, x0)
