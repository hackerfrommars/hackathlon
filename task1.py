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
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
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
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv("Dataset1.csv")
data = data.drop('dateandtime', 1)
cols = data.columns
data = data.replace(' ', np.nan)
data = data.dropna()


for i in range(3):
    data[cols[i]] = data[cols[i]].astype(float)
data[cols[3]] = data[cols[3]].astype(int)

data1 = data[['N', 'EGT', 'WF']].values
label = data['Status'].values

scaler = StandardScaler()
data = scaler.fit_transform(data1)

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)


# neigh = KNeighborsClassifier(n_neighbors = 5)  # 5 better result than most  <5 i think would overfit...
# neigh.fit(x_train, y_train)
# predicted = neigh.predict(x_test)
# print ("accur sc: ",accuracy_score(y_test, predicted))
# # print(average_precision_score(y_test, test_data))
# print ("class_rep: ", classification_report(y_test, predicted))
#
#
# clf = tree.DecisionTreeClassifier()
# clf.fit(x_train, y_train)
# predicted = clf.predict(x_test)
# print ("accur sc: ",accuracy_score(y_test, predicted))
# # print(average_precision_score(y_test, test_data))
# print ("class_rep: ", classification_report(y_test, predicted))
#
# rand_for = RandomForestClassifier(random_state = 0, n_estimators = 10, min_samples_split = 6, min_samples_leaf = 2)
# rand_for.fit(x_train, y_train)
# predicted = rand_for.predict(x_test)
# print ("accur sc: ",accuracy_score(y_test, predicted))
# # print(average_precision_score(y_test, test_data))
# print ("class_rep: ", classification_report(y_test, predicted))

# crange=list(range(1//10, 2, 0.1))
# acc_score=[]
# for c in crange:
#     svc = svm.SVC(kernel = 'linear',  C=c)
#     scores = cross_val_score(svc, data, label, cv=10, scoring='accuracy')
#     acc_score.append(scores.mean())
# print(sum(acc_score) / float(len(acc_score)))


# svm_cl = svm.SVC(kernel='linear', C=)
# svm_cl.fit(x_train, y_train)
# predicted = svm_cl.predict(x_test)
# print ("accur sc: ",accuracy_score(y_test, predicted))
# # print(average_precision_score(y_test, test_data))
# print ("class_rep: ", classification_report(y_test, predicted))


## presumably 3 is better
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=200, random_state=0)
#     kmeans.fit(data)
#     wcss.append(kmeans.inertia_)
#
# plt.plot(wcss)
# plt.xticks(range(1, 11))
# plt.title('The elbow method')
# plt.xlabel('Number of clusters')
# plt.ylabel('wcss score')
# plt.show()



# kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, random_state=0)
# y_means = kmeans.fit_predict(data)


##AgglomerativeClustering
kmeans = AgglomerativeClustering(n_clusters=3, linkage='complete', affinity='euclidean')
y_means = kmeans.fit_predict(data)


plt.scatter(data[y_means==0, 1], data[y_means==0, 2], s=50, c='red', label='Cluster1')
plt.scatter(data[y_means==1, 1], data[y_means==1, 2], s=50, c='blue', label='Cluster2')
plt.scatter(data[y_means==2, 1], data[y_means==2, 2], s=50, c='cyan', label='Cluster3')

# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=70, c='yellow', label='centroids')
plt.title('clusters of materials')
plt.legend()
plt.show()
