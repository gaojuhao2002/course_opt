#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:GJH 
# time:2022/9/22
import numpy as np
class Diagonal_4:
    """Diagonal_4函数
    f=sigma(i=1)(n/2)(0.5*(x[2*i-1]**2+c*(x[2*i]**2)))     其中c=100
     =0.5*(f1 **2+f2**2+...+fm**2)
    % Problem no. 19
    % 维数->  n=variable, 项数->m=2*(n/2)=n;其中n为偶数
    % 默认初始点->   x=np.ones(n)  [1,1,1,1,...1]
    % 极小值点->   未知
    % Revised on 02/09/22 by gaojuhao
    % 下面的jac一级fvec编写的是2*f的值即(f1 **2+f2**2+...+fm**2)
    % 实际得到f和g乘上了0.5
    """
    #初始化
    def __init__(self, n, x=[], *args, **kwargs):
        self.n=n
        if n%2!=0:
            print('输入n应该是偶数')
            return 0
        if x == []:
            x=np.ones(n)
        self.x = np.asarray(x).flatten()
        self.xopt=np.ones(n)
        self.name = '19号函数'

    #函数向量
    def fvec(self,x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        n=self.n
        m=self.n
        fvec=np.zeros(m,)
        for i in range(int(n/2)):
            #本质上是奇数项的平方加上10倍偶数项的平方，所以索引i多加1 x1**2+(10*x2)**2
            #把x的偶数项赋给fvec的偶数项，索引奇数项赋给fvec的奇数项
            fvec[2*i]=x[2*i-1+1]
            fvec[2*i+1]=10*(x[2*i+1]) #这里是因为10**2=100
        return np.asarray(fvec)

    #雅可比行列式（2f的）
    def jac(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        n = self.n
        m = n
        jac = np.zeros((m, n))
        for i in range(int(n/2)):
            jac[2*i,2*i]=1          #对应位置求偏导
            jac[2*i+1,2*i+1]=10
        return np.array(jac)

    #计算函数值
    def f(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        fvec = self.fvec(x)
        return 0.5*(np.linalg.norm(fvec) ** 2) #fvec的二范数乘0.5

    #计算梯度
    def g(self,x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        jac = self.jac(x)
        fvec = self.fvec(x)
        return 0.5*2 * np.dot(jac.T, fvec) #多乘一个0.5是因为2 * np.dot(jac.T, fvec)是计算2*f的梯度

    #黑塞尔矩阵计算
    def G(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        n = self.n
        G = np.zeros((n, n))
        for i in range(int(n / 2)):
            G[2 * i, 2 * i] = 2  #
            G[2 * i + 1, 2 * i + 1] = 200
        return np.array(0.5*G)

