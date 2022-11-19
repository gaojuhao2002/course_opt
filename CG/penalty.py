#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:GJH 
# time:2022/11/18
import numpy as np
class Penalty:
    def __init__(self,n,x=[],alpha=10e-5):
        self.alpha = alpha
        self.n=n
        x=[i for i in range(n)] if x ==[] else x
        self.x = np.asarray(x).flatten()
    def f(self,x=None):
        x = self.x if x is None else np.asarray(x).flatten()
        f1=sum([(x[i]-1)**2 for i in range(self.n)])
        f2=sum([x[i]**2 for i in range(self.n)])-0.25
        return self.alpha*f1+f2**2
    def g(self,x=None):
        x = self.x if x is None else np.asarray(x).flatten()
        f2 = sum([x[i] ** 2 for i in range(self.n)]) - 0.25
        g=[self.alpha*2*(x[i]-1)+2*f2*2*x[i] for i in range(self.n)]
        return np.array(g)
    def G(self,x=None):
        x = self.x if x is None else np.asarray(x).flatten()
        f2 = sum([x[i] ** 2 for i in range(self.n)]) - 0.25
        temp=[self.alpha * 2+4*(2*x[i]*x[i]+f2) for i in range(self.n)]
        return np.diag(temp)