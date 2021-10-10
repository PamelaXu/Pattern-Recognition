# 导入后续需要用到的库文件
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB  # 伯努利模型

# 读取数据并查看
datatrain = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
# print(datatrain.shape)  # 892，12
# print(datatrain.isnull().sum())  #发现age（177）、carbin（687）、embarked（2）有缺失

# 选取数据集特征,去掉不相关特征
datatrain = datatrain.drop(labels=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
print(datatrain.head())

# 去除缺失值
datatrain = datatrain.dropna()

# 属性转化为数值型
datatrain_dummy = pd.get_dummies(datatrain[['Sex', 'Embarked']])

# 编码后和数据拼接
datatrain_conti = pd.DataFrame(datatrain, columns=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
                               index=datatrain.index)
datatrain = datatrain_conti.join(datatrain_dummy)

X_train = datatrain.iloc[:, 1:]
y_train = datatrain.iloc[:, 0]

# 对test文件进行同样处理，去掉几种无用特征
datatest = data_test.drop(labels=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# print(datatest.head())
print(datatest.isnull().sum())  # age、Fare有缺失

datatest = datatest.fillna(datatest.mean()['Age':'Fare'])  # 填补缺失值
print(datatest.isnull().sum())

datatest_dummy = pd.get_dummies(datatest[['Sex', 'Embarked']])

datatest_conti = pd.DataFrame(datatest, columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'], index=datatest.index)
datatest = datatest_conti.join(datatest_dummy)
X_test = datatest

# 标准化
stdsc = StandardScaler()
X_train_conti_std = stdsc.fit_transform(X_train[['Age', 'SibSp', 'Parch', 'Fare']])
X_test_conti_std = stdsc.fit_transform(X_test[['Age', 'SibSp', 'Parch', 'Fare']])
# 将ndarray转为datatrainframe
X_train_conti_std = pd.DataFrame(data=X_train_conti_std, columns=['Age', 'SibSp', 'Parch', 'Fare'], index=X_train.index)
X_test_conti_std = pd.DataFrame(data=X_test_conti_std, columns=['Age', 'SibSp', 'Parch', 'Fare'], index=X_test.index)

# 有序分类变量Pclass
X_train_cat = X_train[['Pclass']]
X_test_cat = X_test[['Pclass']]
# 无序已编码的分类变量
X_train_dummy = X_train[['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
X_test_dummy = X_test[['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
# 拼接为datatrainframe
X_train_set = [X_train_cat, X_train_conti_std, X_train_dummy]
X_test_set = [X_test_cat, X_test_conti_std, X_test_dummy]
X_train = pd.concat(X_train_set, axis=1)
X_test = pd.concat(X_test_set, axis=1)

clf = BernoulliNB()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
data_test['Survived'] = predicted.astype(int)
data_test[['PassengerId', 'Survived']].to_csv('submission.csv', sep=',', index=False)
