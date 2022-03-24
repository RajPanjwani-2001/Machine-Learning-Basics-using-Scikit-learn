from MachineLearning import FeatureEngineering
import pandas as pd
from sklearn.preprocessing import LabelEncoder,PolynomialFeatures
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,StandardScaler

digits = load_digits()

df = digits.data

df = pd.DataFrame(df)

obj = FeatureEngineering()

df = obj.data_handling(df)

x = df
y = digits.target

scaler_m = MinMaxScaler()
x = scaler_m.fit_transform(x)

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)

lr = LogisticRegression()
lr.fit(X_train,Y_train)

y_pred = lr.predict(X_test)

from sklearn.metrics import accuracy_score 
print(accuracy_score(Y_test,y_pred))

cm = confusion_matrix(Y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.show()