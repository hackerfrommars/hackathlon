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


# data = data[data.Gas_Flow > 4000]
# data = data.loc[(data['Gas_Flow'] < 9800) | (data['Gas_Flow'] > 11400)]

# data.hist()
print(data)
label = data['Gas_Flow'].values
data.drop(['Gas_Flow'],axis=1,inplace=True)
data = data['Almaty'].values.reshape(-1,1)



x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
model = SVR()

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


# pd.DataFrame(y_train).hist()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
N = x_test.shape[0]
print(mean_squared_error(y_test,y_pred)**0.5)
# plt.plot(range(1,N+1),y_test,color='red')
# plt.plot(range(1,N+1),y_pred,color='blue')
plt.show()
