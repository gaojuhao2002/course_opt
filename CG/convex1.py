#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:GJH 
# time:2022/11/18
import numpy as np
class Convex1:
    def __init__(self,n,x=[]):
        self.n = n
        x = [i / n for i in range(n)] if x==[] else x
        self.x=np.array(x).flatten()
    def f(self,x=None):
        x=self.x if x is None else np.asarray(x).flatten()
        f=sum([np.exp(x[i])-x[i] for i in range(self.n)])
        return f
    def g(self,x=None):
        x = self.x if x is None else np.asarray(x).flatten()
        g=[np.exp(x[i])-1 for i in range(self.n)]
        return np.array(g)
    def G(self,x=None):
        x = self.x if x is None else np.asarray(x).flatten()
        #只有对角线位置有数值
        temp=[np.exp(x[i]) for i in range(self.n)]
        return np.diag(temp)
