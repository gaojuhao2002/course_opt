#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:GJH 
# time:2022/9/22

import numpy as np

import diagonal_4

def armijo_backtrack(fun, x0, d0, g0, f0=None, c1=1e-4, alpha0=1, *args, **kwargs):
    """
    利用armijo搜索计算步长: f(xk+1)<= f(xk)+sigma1 * alpha * gk * dk
    输入参数
    ----------
    fun : callable          目标函数
    x0 : array_like         当前迭代点
    d0 : array_like         搜索方向.
    g0 : array_like         目标函数在当前点的梯度.
    f0 : float              当前点的函数值.
    c1: armijo              搜索中的sigma参数
    alpha0:                 初始步长
    args : tuple, optional  计算目标函数f(x)需要的可选参数.
    返回参数：
    -------
    alpha：                  步长
    fc：                     函数值的计算次数
    f1：f(x_(k+1))
"""
    #初始参数设置
    lou=0.5             #压缩因子
    times=1000           #压缩最大次数
    x0 = np.asarray(x0).flatten()
    d0 = np.asarray(d0).flatten()
    g0 = np.asarray(g0).flatten()

    #fc是次数
    for fc in range(times):
        alpha=alpha0*(lou**fc) #fc等于0的时候即alpha0
        f1=fun(x0+alpha*d0)
        if f1<=(f0+c1*alpha*np.dot(g0,d0.T)):
            return alpha,fc+1,f1

#测试  最速梯度下降
test=diagonal_4.Diagonal_4(6)
alpha, fc, f1 = armijo_backtrack(fun=test.f,x0=test.x,d0=-test.g(),g0=test.g(),f0=test.f())
print('测试diagonal_4一次armijo线性搜索的alpha为{}\nfc为{}\nf1为{}\n'.format(alpha,fc,f1))

#测试给定二次函数
def fun2(x):
    x = np.asarray(x).flatten()
    return x[0] ** 2 + 2 * x[1] ** 2


def gfun(x):
    x = np.asarray(x).flatten()
    g = np.zeros([2, 1])
    g[0] = 2 * x[0]
    g[1] = 4 * x[1]
    return g
alpha, fc, f1 = armijo_backtrack(fun=fun2,x0=np.asarray([1,0]),d0=np.asarray([-1,-1]),g0=gfun(np.asarray([1,0])),f0=fun2(np.asarray([1,0])))
print('二次函数f = x**2 +2*y**2在(1, 0)处沿方向（-1，-1）满足armijo条件的步长alpha为{}\nfc为{}\nf1为{}\n'.format(alpha,fc,f1))

#获取最优解测试
yopt=diagonal_4.Diagonal_4(10)   #创建优化函数对象
iter=1000                       #循环次数

fun=yopt.f                      #初始化参数
x=yopt.x
d=-yopt.g()
g=yopt.g()
f=yopt.f()
epsilon =1e-8
                                #循环求解最优解（最速梯度下降）
for i in range(iter):
    if np.linalg.norm(g) ** 2>epsilon:
        alpha, fc, f1 = armijo_backtrack(fun, x, d, g, f)
        f=f1
        x=x+alpha*d
        g=yopt.g(x)
        d=-g
    else:
        break
print("利用负梯度armijo线性搜索方法求解函数的最优解为{}\n此时x取值{}".format(f,x))
print('最后一次迭代的alpha为{}\nfc为{}\nf1为{}\n'.format(alpha, fc, f1))






