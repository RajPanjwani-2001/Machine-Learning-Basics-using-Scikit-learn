from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('income.csv')

data = df.values
print(df)

x = data[:,1:-1]
y = data[:,-1]

plt.scatter(x,y)
plt.show()

km= KMeans(n_clusters=3)
y_pred = km.fit_predict(x,y)
print(y_pred)
y_pred = y_pred[:,np.newaxis]
print(y_pred.shape)
#df.iloc[:,-1] = y_pred
#np.hstack([data, y_pred])
np.append(x,y_pred,axis=1)
print(x)

new_x = x[:,1:-1]
print(new_x)
new_y = x[:,-1]
print(new_y)

plt.scatter(new_x,new_y)
plt.show()

