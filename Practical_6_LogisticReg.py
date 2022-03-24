from MachineLearning import FeatureEngineering
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('insurance.csv')
obj = FeatureEngineering()

df = obj.data_handling(df)
data = df.values

x = data[:,:-1]
y = data[:,-1]

model = LogisticRegression()
model.fit(x,y)

ac,y_pred = obj.calc_acc(model,x,y)
print(ac)
