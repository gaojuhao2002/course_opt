#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:GJH
# time:2022/10/31
# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author:GJH
# time:2022/9/2
import numpy as np


class Extend_Rosenbrock:
    def __init__(self, n, x=[], *args, **kwargs):
        self.n = n
        if n % 2 != 0:
            print('输入n应该是偶数')
            return 0
        self.m = n  # m=n*2/2
        if x == []:
            for i in range(n):
                if i % 2 == 0:
                    x.append(-1.2)
                else:
                    x.append(1)
        self.x = np.asarray(x).flatten()  # 转化为一维数组
        assert self.x.shape[0] == self.n,'自变量维度有误'
        self.xopt = np.ones(n)  # 假设已知最优结果
        self.name='RosenBrock'
    def fvec(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        m=self.m
        n=self.n
        fvec = np.zeros((m,))
        for i in range(int(n / 2)):
            fvec[2 * i] = (10 * (x[2 * i + 1] - x[2 * i] ** 2))
            fvec[2 * i + 1] = (1 - x[2 * i])

        return np.array(fvec)

    # 雅各比行列式
    def jac(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        jac = np.zeros((self.m, self.n))
        for i in range(int(self.n / 2)):
            jac[2 * i, 2 * i + 1] = 10
            jac[2 * i, 2 * i] = -20 * x[2 * i]
            jac[2 * i + 1, 2 * i] = -1
        return np.array(jac)

    def f(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        fvec = self.fvec(x)
        return np.linalg.norm(fvec) ** 2

    def g(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        jac = self.jac(x)
        fvec = self.fvec(x)
        return 2 * np.dot(jac.T, fvec)

    def G(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        n = self.n
        G = np.zeros((n, n))
        for i in range(int(n / 2)):
            G[2 * i + 1, 2 * i + 1] = 200
            G[2 * i + 1, 2 * i] = -400*x[2*i]
            G[2 * i , 2 * i]=1200*x[2*i]*x[2*i]-400*x[2 * i+1]+2
            G[2 * i, 2 * i + 1]=-400*x[2*i]
        return G