import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

import sklearn.metrics as skmet
import warnings
warnings.filterwarnings("ignore")

cancerdata = pd.read_csv("cancerdata.csv")

# Data Preprocessing & EDA
# Converting B to Benign and M to Malignant 
cancerdata['diagnosis'] = np.where(cancerdata['diagnosis'] == 'B', 'Benign', cancerdata['diagnosis'])
cancerdata['diagnosis'] = np.where(cancerdata['diagnosis'] == 'M', 'Malignant', cancerdata['diagnosis'])

cancerdata.drop(['id'], axis = 1, inplace = True)
cancerdata.info()

cancerdata.describe()

cancerdata_X = pd.DataFrame(cancerdata.iloc[:, 1:])
cancerdata_y = pd.DataFrame(cancerdata.iloc[:, 0])

numeric_features = cancerdata_X.select_dtypes(exclude = ['object']).columns

num_pipeline = Pipeline([('impute', SimpleImputer(strategy = 'median'))])

categorical_features = cancerdata_X.select_dtypes(include = ['object']).columns

categ_pipeline = Pipeline([('label', DataFrameMapper([(categorical_features, OneHotEncoder(drop = 'first'))]))])


preprocess_pipeline = ColumnTransformer([('categorical', categ_pipeline, categorical_features),('numerical', num_pipeline, numeric_features)])

processed = preprocess_pipeline.fit(cancerdata_X)

cancerclean = pd.DataFrame(processed.transform(cancerdata_X), columns = cancerdata_X.columns)  # Cleaned and processed data for ML Algorithm

cancerclean.info()

scale_pipeline = Pipeline([('scale', MinMaxScaler())])

preprocess_pipeline2 = ColumnTransformer([('scale', scale_pipeline, cancerclean.columns)]) 

processed2 = preprocess_pipeline2.fit(cancerclean)

cancerclean_n = pd.DataFrame(processed2.transform(cancerclean), columns = cancerclean.columns)

Y = np.array(cancerdata_y['diagnosis']) # Target

X_train, X_test, Y_train, Y_test = train_test_split(cancerclean_n, Y, test_size = 0.2, random_state = 0)

X_train.shape
X_test.shape

acc = []

for i in range(3, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    diff = train_acc - test_acc
    acc.append([diff, train_acc, test_acc])
acc

# Plotting the data accuracies
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], "ro-")
plt.plot(np.arange(3, 50, 2), [i[2] for i in acc], "bo-")

# Fit the model
knn = KNeighborsClassifier(n_neighbors = 9)
KNN = knn.fit(X_train, Y_train) 

# Predict the class on test data
pred_test = KNN.predict(X_test)

skmet.accuracy_score(Y_test, pred_test)
pd.crosstab(Y_test, pred_test, rownames = ['Actual'], colnames = ['Predictions']) 

