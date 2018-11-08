#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Xiang

from sklearn import preprocessing
import numpy as np

X = np.array([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])
X_scaled = preprocessing.scale(X)

print("========X==========")
print(X)
print("========X_scaled==========")
print(X_scaled)
print("========mean==========")
# scale函数可实现数据的均值为0，方差为1
print(X_scaled.mean(axis=0))  # 按列计算均值
print("========var==========")
print(X_scaled.var(axis=0))  # 按列计算方差

# 运行结果：
'''
== == == == X == == == == ==
[[1. - 1.  2.]
 [2.  0.  0.]
[0.
1. - 1.]]
== == == == X_scaled == == == == ==
[[0. - 1.22474487  1.33630621]
 [1.22474487  0. - 0.26726124]
[-1.22474487
1.22474487 - 1.06904497]]
== == == == mean == == == == ==
[0.  0.  0.]
== == == == var == == == == ==
[1.  1.  1.]
'''