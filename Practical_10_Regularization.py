from MachineLearning import FeatureEngineering
import pandas as pd
from sklearn.preprocessing import LabelEncoder,PolynomialFeatures
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression,Lasso,Ridge

df = pd.read_csv('Melbourne_housing_FULL.csv')
obj = FeatureEngineering()

df = df.dropna()
print(df.isna().sum())

df = pd.get_dummies(df,drop_first=True)

print(df)

data = df.values

x = data[:,:-1]
y = data[:,-1]

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#lr = LinearRegression()
#lr = Lasso(alpha=50,max_iter=100,tol=0.1)
lr = Ridge(alpha=50,max_iter=100,tol=0.1)
lr.fit(X_train,Y_train)

y_pred = lr.predict(X_test)
y_pred = np.round(y_pred)
ac = accuracy_score(Y_test,y_pred)
print(ac)
