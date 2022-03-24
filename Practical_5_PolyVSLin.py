from MachineLearning import FeatureEngineering
import pandas as pd
from sklearn.preprocessing import LabelEncoder,PolynomialFeatures
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('winequality-red.csv')
obj = FeatureEngineering()

df = obj.data_handling(df)
data = df.values


x = data[:,:-1]
y = data[:,-1]

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(X_train.shape)
model = LinearRegression()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)
ac_lr = accuracy_score(Y_test,y_pred)
print('accuracy_score of linear regression: ', ac_lr)

poly_feat = PolynomialFeatures(degree=2)
x_poly = poly_feat.fit_transform(x)
X_train,X_test,Y_train,Y_test = train_test_split(x_poly,y,test_size=0.2,random_state=0)

lr = LinearRegression()
lr.fit(X_train,Y_train)
y_pred = lr.predict(X_test)
y_pred = np.round(y_pred)
ac_poly = accuracy_score(Y_test,y_pred)
print('accuracy_score of polynomial regression: ', ac_poly)

if ac_lr>ac_poly:
    print('LinearRegression has more accuracy')
else:
    print('PolyRegression has more accuracy')
    
