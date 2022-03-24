from sklearn.svm import SVC
from MachineLearning import FeatureEngineering
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('heart.csv')
obj = FeatureEngineering()

df = obj.data_handling(df)
data = df.values

x = data[:,:-1]
y = data[:,-1]

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)

model = SVC()
model.fit(X_train,Y_train)

ac,y_pred = obj.calc_acc(model,X_test,Y_test)
print('Accuracy before Standard Scaler: ',ac)

scaler_m = StandardScaler()
x = scaler_m.fit_transform(x)

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
model = SVC()
model.fit(X_train,Y_train)

ac,y_pred = obj.calc_acc(model,X_test,Y_test)
print('Accuracy after Standard Scaler: ',ac)