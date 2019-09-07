# -*- coding: utf-8 -*- 
# @Author : yunze


"""
sklearn建模四步骤
1、调用需要使用的模型类
2、模型初始化
3、模型训练
4、模型预测
"""
# 调用需要的模型类
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
#
# # iris数据（花瓣的长宽）
iris = datasets.load_iris()
x = iris.data
#  目标数据，也就是说是数据量化，这里量化成0，1，2
y = iris.target
print(iris.data)


print(iris.target)
# 模型初始化
knn = KNeighborsClassifier(n_neighbors=1)

# 模型训练
knn.fit(x, y)

# 模型预测
x_test = [[1, 2, 3, 4]]
print(knn.predict(x_test))
