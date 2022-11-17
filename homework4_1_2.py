#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:GJH 
# time:2022/10/17
import numpy as np
import homework4_1_1
import matplotlib.pyplot as plt
def acmaxdownwithx(getf, getg, G, x, epsilon=1e-6,x_k=[],iter=1000):
    #x_k用于保存迭代轨迹
    x = np.asarray(x)
    x_k.append(x)
    G = np.asarray(G)

    for k in range(iter):
        g = getg(x)  # 计算梯度此时g为行向量

        # 如果终止条件满足直接结束(满足g的范数低于给定值)
        if np.linalg.norm(g) < epsilon:
            print('\n89页习题2的函数查找第{}次发现,函数最优解为{}，函数的最优值为{},'
                  '此时梯度的范数为{}'.format(k,x, getf(x),np.linalg.norm(g)))
            return x_k

        # 若不满足最优条件
        d = -g                                      # 计算负梯度方向
        alpha = np.dot(g, g.T) / np.dot(g, np.dot(G,g.T))  # 计算步长
        x = x + alpha * d                           # 更新迭代点
        if k == iter-1:
            print("经过了{}次搜索没有找到满足终止条件的最优值".format(iter))
            return x_k
        x_k.append(x)                               #添加到迭代点序列
def x_n(n):
    #得到n个题目的点列
    temp=[]
    for k in range(n):
        x=np.asarray([0,1-(1/(5**k))])
        temp.append(x)
    return np.asarray(temp)
if __name__ == '__main__':
    # 89页习题2
    print('89页习题2求解')
    G = np.array([[4, -2], [-2, 2]])
    b = np.array([2, -2])
    x0 = np.array([0, 0])
    sub = homework4_1_1.Square_function(x0, G, b, 0)
    x_k=acmaxdownwithx(sub.f,sub.g,G,x0)
    x_k=np.asarray(x_k)
    x_odd=x_n(int(0.5*len(x_k)))


    plt.plot(x_k[:,0],x_k[:,1])
    plt.scatter(x_odd[:,0],x_odd[:,1],c='r')
    plt.title('predict argmin=(0,1)')
    plt.legend(['x_k_trace','x_odd'])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

