# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#1获取数据
#1.1导入sklearn数据集
from sklearn import datasets
iris=datasets.load_iris()#导入分类算法花数据集（load_boston()波士顿房价回归集；load_digits()手写数字分类集）
x=iris.data #获取特征向量
y=iris.target #获取样本label

#1.2导入文件数据
path = "../input/"
train = pd.read_csv(path+"train.csv")
train.columns=['q','w','e','r','label']
test = pd.read_csv(path+"data_iris_test.csv", header=-1)
test.columns=['q','w','e','r']
print(train.head(10))
print(test.head(10)) #显示前十个
print(train.describe()) #描述

#1.3创建数据集
from sklearn.datasets.samples_generator import make_classification
X,Y=make_classification(n_samples=6,n_features=5,n_informative=2,n_redundant=2,n_classes=2,n_clusters_per_class=2,scale=1.0,random_state=20)
#n_samples:制定样本数
#n_features:指定特征数
#n_classes:指定几分类
#random_state:随机种子，使得随机状可重
for x_,y_ in zip(X,Y):
    print(y_)
    print(x_)

#2.数据预处理
#2.1特殊值替换
train['q'] = train['q'].replace(510000000, 67000) #值的替换
print(train)
plt.rcParams['font.sans-serif']=['simHei']#用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.plot(train.q)
plt.title('标题',fontsize=30)
plt.xlabel(u'我是横坐标',fontsize=5)
plt.ylabel(u'我是纵坐标')
plt.show()
plt.close()
# plt.plot(train.w)
# plt.show()
# plt.close()
# plt.plot(range(0, train.shape[0]), train.e)
# plt.show()
# plt.close()
# plt.plot(train.r)
# plt.show()
# plt.close()

#2.2缺失值填充
train['q'] = train.q.fillna(train['q'].median())
train['w'] = train.w.fillna(train['w'].mean())
train['e'] = train.e.fillna(train['e'].mean())
train['r'] = train.r.fillna(train['r'].mean())
print(train.label.value_counts())

# sns.set()
# sns.pairplot(train, hue='label')
# plt.show()

#2.3训练值和测试值合并
train['type'] = 1
test['type'] = 0
val = train['label']
print("--------合并前打印train-----------")
print(train)
print("----------合并前打印test----------")
print(test)
all_data = pd.concat([train.drop(['label'], axis=1), test]) #将两个data合并。
print("===================alldata==================")
print(all_data)
print("============================================")

# 2.4导入数据归一化
from sklearn.preprocessing import scale
all_data['q'] = scale(all_data['q'])
all_data['w'] = scale(all_data['w'])
all_data['e'] = scale(all_data['e'])
all_data['r'] = scale(all_data['r'])
print('相关性分析：',all_data.corr()) #相关性显示

#2.5sklearn数据集归一化
from sklearn import preprocessing

#a.基于mean和std的标准化
#scaler=preprocessing.StandardScaler(),fit(train_data)
#scaler.transform(train_data)
#scaler.transform(test_data)
#print('train_data:',train_data)
#print('scaler:',scaler)

#b.将每个特征值归一化到一个固定范围
#scaler=preprocessing.MinMaxScaler(feature_range=(0,1)).fit(train_data)
#scaler.transform(train_data)
#scaler.transform(test_data)
##feature_range:定义归一化范围，注用()括起来

#2.6正则化preprocessing.normalize(X,norm='12')
#2.7one-hot编码preprocessing.OneHotEncoder().fit(data)
   #one-hot编码是一种对离散特征值的编码方式，在LR模型中常用到，用于给线性模型增加非线性能力

#3.数据集拆分
    # 作用：将数据集划分为 训练集和测试集
train = all_data[all_data['type'] == 1]
test = all_data[all_data['type'] == 0]
train = train.drop(['type'], axis=1)
test = test.drop(['type'], axis=1)
print('训练集拆分前：',train)
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(train, val, train_size=0.8, random_state=42) #把train分为训练集和验证集
print('打印验证集val_y原结果：',val_y)

#4.定义模型
#4.1 支持向量机svm的调用
import sklearn.svm as svm
svc = svm.SVC()
#svc=svm.SVC(C=1.0,kernel='rbf',gamma='auto')
"""参数
    C：误差项的惩罚参数C
    gamma: 核相关系数。浮点数，If gamma is ‘auto’ then 1/n_features will be used instead.
"""
svc.fit(train_X, train_y) #拟合模型
preds_svc = svc.predict(val_X) #模型预测
print('svm预测结果：',preds_svc)

#4.2 决策树DT的调用
import sklearn.tree as tree
dc = tree.DecisionTreeClassifier()
#dc=tree.DecisionTreeClassifier(criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,class_weight=None, presort=False)
"""参数
    criterion ：特征选择准则gini/entropy
    max_depth：树的最大深度，None-尽量下分
    min_samples_split：分裂内部节点，所需要的最小样本树
    min_samples_leaf：叶子节点所需要的最小样本数
    max_features: 寻找最优分割点时的最大特征数
    max_leaf_nodes：优先增长到最大叶子节点数
    min_impurity_decrease：如果这种分离导致杂质的减少大于或等于这个值，则节点将被拆分。
"""
dc.fit(train_X, train_y)
preds_dc = dc.predict(val_X)
print('tree预测结果：',preds_dc)

#4.3 线性回归
from sklearn.linear_model import LinearRegression
#lir=LinearRegression(fit_intercept=True,normalize=False,copy_X=True,n_jobs=1)
lir=LinearRegression()
"""参数
    fit_intercept：是否计算截距。False-模型没有截距
    normalize： 当fit_intercept设置为False时，该参数将被忽略。 如果为真，则回归前的回归系数X将通过减去平均值并除以l2-范数而归一化。
     n_jobs：指定线程数
     y=ax+b(a:model.coef_; b:moddel.intercept_)
"""
lir.fit(train_X,train_y)
preds_lir=lir.predict(val_X)
print('线性回归预测结果：',preds_lir)

#4.4逻辑回归LR
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
    fit_intercept=True, intercept_scaling=1, class_weight=None,
    random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
    verbose=0, warm_start=False, n_jobs=1)
"""参数
    penalty：使用指定正则化项（默认：l2）
    dual: n_samples > n_features取False（默认）
    C：正则化强度的反，值越小正则化强度越大
    n_jobs: 指定线程数
    random_state：随机数生成器
    fit_intercept: 是否需要常量
"""
lr.fit(train_X,train_y)
preds_lr=lr.predict(val_X)
print('逻辑回归LR预测结果：',preds_lr)

#4.5 朴素贝叶斯算法NB
from sklearn import naive_bayes
nbs=naive_bayes.GaussianNB() #高斯贝叶斯
#nbs = naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
#nbs = naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
"""
文本分类问题常用MultinomialNB
参数
    alpha：平滑参数
    fit_prior：是否要学习类的先验概率；false-使用统一的先验概率
    class_prior: 是否指定类的先验概率；若指定则不能根据参数调整
    binarize: 二值化的阈值，若为None，则假设输入由二进制向量组成
"""
nbs.fit(train_X,train_y)
preds_nbs=nbs.predict(val_X)
print('高斯算法NB预测结果：',preds_nbs)

#4.6 k近邻算法KNN
from sklearn import neighbors
knn=neighbors.KNeighborsClassifier(n_neighbors=5,n_jobs=1)#分类
#knn=neighbors.KNeighborsRegressor()n_neighbors=5,n_jobs=1) #回归
"""参数
    n_neighbors： 使用邻居的数目
    n_jobs：并行任务数
"""
knn.fit(train_X,train_y)
preds_knn=knn.predict(val_X)
print('k近邻算法KNN预测结果：',preds_knn)

#4.7 多层感知机(神经网络)
from sklearn.neural_network import MLPClassifier
mlpc=MLPClassifier()
#mlpc=MLPClassifier(activation='relu',solver='adam',alpha=0.0001)
"""参数
    hidden_layer_sizes: 元祖
    activation：激活函数
    solver ：优化算法{‘lbfgs’, ‘sgd’, ‘adam’}
    alpha：L2惩罚(正则化项)参数。
"""
mlpc.fit(train_X,train_y)
preds_mlpc=mlpc.predict(val_X)
print('多层感知机预测结果：',preds_mlpc)

#------5 模型评估函数---------------
#5.1 accuracy_score模型评估
from sklearn.metrics import accuracy_score
def eval_class(preds, truth):
    return accuracy_score(preds, truth)
print('4.1的支持向量机svc模型评估：',eval_class(preds_svc, val_y))
print('4.2的决策树：',eval_class(preds_dc, val_y))
#print('4.3的线性回归：',eval_class(preds_lir,val_y))
print('4.4的逻辑回归LR：',eval_class(preds_lr,val_y))
print('4.5的朴素贝叶斯：',eval_class(preds_nbs,val_y))
print('4.6的近邻算法KNN：',eval_class(preds_knn,val_y))
print('4.7的多层感知机：',eval_class(preds_mlpc,val_y))

##5.2交叉验证
#from sklearn.model_selection import cross_val_score
#cross_val_score(svc,X,y=None,scoring=None,cv=None,n_jobs=1)
"""参数
    model：拟合数据的模型
    cv ： k-fold
    scoring: 打分参数-‘accuracy’、‘f1’、‘precision’、‘recall’ 、‘roc_auc’、'neg_log_loss'等等
"""

#5.3检验曲线
from sklearn.model_selection import validation_curve
train_score, test_score = validation_curve(svc, X, y, param_name, param_range, cv=None, scoring=None, n_jobs=1)
"""参数
    model:用于fit和predict的对象
    X, y: 训练集的特征和标签
    param_name：将被改变的参数的名字
    param_range： 参数的改变范围
    cv：k-fold

返回值
---
   train_score: 训练集得分（array）
    test_score: 验证集得分（array）
"""

#6.保存模型
#6.1保存为pickle文件
import pickle
#保存模型
with open('model,pickle','wb') as f:
    pickle.dump(model,f)
#读取模型
with open(model.pickle,'rb') as f:
    model=pickle.load(f)
model.predict(val_X)

#6.2sklearn自带方法joblib
from sklearn.externals import joblib
#保存模型
joblib.dump(model,'model.pickle')
#载入模型
model=joblib.load('model.pickle')

#res = pd.DataFrame(dc.predict(test))
#res.to_csv("preds_iris.csv", index=False)