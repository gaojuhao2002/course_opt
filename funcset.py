#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:GJH 
# time:2022/8/26

import numpy as np
a=np.array([[1,2,3,4],[5,6,7,8]])          #ndmin 设置最小维度
print(a)
print(type(a))
print(a.shape)                              #a.sahape属性查看数组规模
b=a.reshape(4,2)                            #a.reshape()方法重构规模 —先拉伸为一维
print(b)

