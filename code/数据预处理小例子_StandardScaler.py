# 2.StandardScaler
# -*- coding:utf-8 -*-
from sklearn.preprocessing import StandardScaler
import numpy as np

X1 = [[1, 2, 3, 2],
      [4, 5, 7, 9],
      [8, 7, 4, 3],
      [5, 9, 4, 2],
      [1, 4, 7, 8]]

ss = StandardScaler()  # 引入标准化的方法，要有数据的均值和方差
ss.fit(X1)  # 用标准化的方法进行训练数据集X1（fit），得到标准化方法中的参数
# ss = StandardScaler.fit(X1)   等价于上面两句
print(ss)
print("========mean_==========")
print(ss.mean_)
print("========var_==========")
print(ss.var_)

print("===transform实现标准化转换===")
print(ss.transform(X1))

# fit与transform结合成一句
print("===fit_transform训练并标准化转换===")
X1_train = ss.fit_transform(X1)
print(X1_train)
#X=np.tolist(X1_train)
X=np.tolist
print(X)

# 运行结果：
'''
StandardScaler(copy=True, with_mean=True, with_std=True)
== == == == mean_ == == == == ==
[3.8  5.4  5.   4.8]
== == == == var_ == == == == ==
[6.96  5.84  2.8   9.36]
== =transform实现标准化转换 == =
[[-1.06133726 - 1.40693001 - 1.19522861 - 0.91520863]
 [0.0758098 - 0.16552118  1.19522861  1.37281295]
[1.59200589
0.66208471 - 0.5976143 - 0.58834841]
[0.45485883  1.4896906 - 0.5976143 - 0.91520863]
[-1.06133726 - 0.57932412
1.19522861
1.04595272]]
== =fit_transform训练并标准化转换 == =
[[-1.06133726 - 1.40693001 - 1.19522861 - 0.91520863]
 [0.0758098 - 0.16552118  1.19522861  1.37281295]
[1.59200589
0.66208471 - 0.5976143 - 0.58834841]
[0.45485883  1.4896906 - 0.5976143 - 0.91520863]
[-1.06133726 - 0.57932412
1.19522861
1.04595272]]
'''