from MachineLearning import FeatureEngineering
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC

df = pd.read_csv('iris.csv')
obj = FeatureEngineering()

df = obj.data_handling(df)
data = df.values

x = data[:,:-1]
y = data[:,-1]

x_pca = obj.pca(x,2)

X_train,X_test,Y_train,Y_test = train_test_split(x_pca,y,test_size=0.2,random_state=0)

model = SVC()
model.fit(X_train,Y_train)

ac,y_pred = obj.calc_acc(model,X_test,Y_test)
print(ac)

cm = confusion_matrix(Y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.show()