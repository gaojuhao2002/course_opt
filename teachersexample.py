#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:GJH 
# time:2022/9/2
import numpy as np
class Rosen_g:
    """广义的Rosenbrock函数
    f = sigma(i = 1)(n -1)(100.0*(x[i + 1] - x[i]**2.0)**2.0 + (1 - x[i])**2.0)
         =  (f1 ** 2+ f2 ** 2+ ... +f2n-2 ** 2)
    % Problem no. 1
    % 维数      ->  n=variable, m=2*n/2=n; m即项数当且仅当多项式问题时出现
    % 默认初始点 -> x=(s(j))，其中         如果没有给出用空属性None或者[]，未来再添加
    %                s(2*j-1)=-1.2,
    %                s(2*j)=1
    % 极小值点 -> f=0 at (1,.....,1)
    % Revised on 03/22/2021 by zhangjw
    """

    def __init__(self, n=6, x=[], *args, **kwargs):
        self.n = n
        # x = []
        # 如果没有初始点自己赋初始
        if x == []:
            for i in range(n):
                if i % 2 == 0:
                    x.append(-1.2)
                else:
                    x.append(1)
        self.x = np.asarray(x).flatten()  # 转化为一维数组
        self.xopt = np.ones(n)  # 假设已知最优结果

    def fvec(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        n = self.n
        m = 2 * (n - 1)
        fvec = np.zeros((m,))
        for i in range(n - 1):
            fvec[2 * i] = (10 * (x[i + 1] - x[i] ** 2))
            fvec[2 * i + 1] = (1 - x[i])
        return np.array(fvec)
    #雅各比行列式
    def jac(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        n = self.n
        m = 2 * (n - 1)
        jac = np.zeros((m, n))
        for i in range(n - 1):
            jac[2 * i, i] = -20 * x[i]
            jac[2 * i, i + 1] = 10
            jac[2 * i + 1, i] = -1
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
        pass


# if __name__ == 'teachersexample':
f1 = Rosen_g(6)
print(f1.x,f1.fvec(),f1.jac(),f1.f(),f1.g())