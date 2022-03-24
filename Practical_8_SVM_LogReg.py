from MachineLearning import FeatureEngineering
import pandas as pd
from sklearn.preprocessing import LabelEncoder,PolynomialFeatures
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

cancer = load_breast_cancer()

df = cancer.data
df = pd.DataFrame(df)
obj = FeatureEngineering()

df = obj.data_handling(df)

x = df
y = cancer.target

print(x)


X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)

lr = LogisticRegression()
lr.fit(X_train,Y_train)

lr_ac,y_pred = obj.calc_acc(lr,X_test,Y_test)
print('Logistic Regression Accuracy: ',lr_ac)

cm = confusion_matrix(Y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.show()

svc = SVC()
svc.fit(X_train,Y_train)

svc_ac,y_pred = obj.calc_acc(svc,X_test,Y_test)
print('SVM Accuracy: ',svc_ac)

cm = confusion_matrix(Y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.show()