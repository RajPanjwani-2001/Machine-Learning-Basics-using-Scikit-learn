from MachineLearning import FeatureEngineering
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

obj = FeatureEngineering()
df_train = pd.read_csv('train.csv')
df_train = obj.data_handling(df_train)
data = df_train.values

X_train = data[:,:-1]
Y_train = data[:,-1]

df_test = pd.read_csv('train.csv')
df_test = obj.data_handling(df_test)
data = df_test.values

X_test = data[:,:-1]
Y_test = data[:,-1]

model = DecisionTreeClassifier()
model.fit(X_train,Y_train)

ac,y_pred = obj.calc_acc(model,X_test,Y_test)
print('DecisionTree Accuracy : ',ac)

model = RandomForestClassifier()
model.fit(X_train,Y_train)

ac,y_pred = obj.calc_acc(model,X_test,Y_test)
print('RandomForest Accuracy : ',ac)