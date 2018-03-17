import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV

#other libraries
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
import pandas as pd
import seaborn as sns


#metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#feature selection
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC, SVR
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv("Dataset2.csv")
data = data.drop(data.columns[[0, 4, 6]], 1) # removing (dateandtime, Astana, Oskemen)
data = data.dropna() #removing NaN records
shape = data.shape #after removing NaN records

cols = data.columns
for i in range(shape[1]):
    data[cols[i]] = data[cols[i]].astype(float)

def heatmap(fulldata,figsize=(25,25),annot_size = 8,cmap=sns.cubehelix_palette(start = 0.2,rot = 0.3,dark = 0.15,light = 0.85,as_cmap = True)):
    corr = fulldata.corr()
    _,ax = plt.subplots(1,1,figsize=figsize)
    sns.heatmap(corr,
    cbar=True,
    cbar_kws={'shrink':0.9},
    annot=True,
    annot_kws={'fontsize':annot_size},
    cmap = cmap
    )
    plt.show()
# heatmap(data)

# data = data.drop(data.columns[[3]], 1) # removing Atyrau because correlation is very low -0.24

data1 = data[['Almaty', 'Kyzylorda', 'Atyrau']].values
data_aty = data['Atyrau'].values
data_alm = data['Almaty'].values
label = data['Gas_Flow'].values
# plt.subplot(211)
# plt.scatter(data_alm, label, c='red', alpha=0.5)
# plt.subplot(212)
# plt.scatter(data_kyz, label, c='blue', alpha=0.58)
# plt.show()

data_aty = data_aty.reshape(-1, 1)
data_alm = data_alm.reshape(-1, 1)
scaler = StandardScaler()
data_alm = scaler.fit_transform(data_alm)

x_train, x_test, y_train, y_test = train_test_split(data_alm, label, test_size=0.2, random_state=42)

nn = MLPRegressor(activation='relu', )
n = nn.fit(x_train, y_train)
predicted = nn.predict(x_test)
# print('regression coef: ', log_reg.coef_)
print("mean squeared error: ", mean_squared_error(y_test, predicted))



# log_reg = LogisticRegression(C=1e5)
# log_reg.fit(x_train, y_train)
# predicted = log_reg.predict(x_test)
# # print('regression coef: ', log_reg.coef_)
# print("mean squeared error: ", mean_squared_error(y_test, predicted))

# svm_reg = SVR(kernel='rbf', C=1.00)
# svm_reg.fit(x_train, y_train)
# predicted = svm_reg.predict(x_test)
# print("mean squeared error: ", mean_squared_error(y_test, predicted))


# rand_for = RandomForestRegressor()
# rand_for.fit(x_train, y_train)
# print(rand_for.feature_importances_) # feature importances by cities
#
# predicted = rand_for.predict(x_test)
# print("mean squeared error: ", mean_squared_error(y_test, predicted))


# regr = LinearRegression()
# regr.fit(x_train, y_train)
# predicted = regr.predict(x_test)
# print('regression coef: ', regr.coef_)
# print("mean squeared error: ", mean_squared_error(y_test, predicted))
