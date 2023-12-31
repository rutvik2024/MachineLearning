# -*- coding: utf-8 -*-
"""Que5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_gWKx659oBbWq8ATS69pwbzZUNZJ8ED8
"""

# Importing Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""## 5.1 Plot a graph for sigmoid function."""

x_axis = np.linspace(-10, 10, 100)
y_axis = 1/(1+np.exp(-x_axis))

plt.plot(x_axis,y_axis)
plt.show()

"""## 5.2 Experiment 2: Build a linear classifier."""

data = pd.read_csv('Que5.csv')
data.head()
classes = data['Classes  '].to_numpy()
temp = []
for i in classes:
    if i == 'not fire   ':
      temp.append(0)
    else:
        temp.append(1)
data['Classes  '] = np.array(temp)
# data
isi = data['DC'].to_numpy()
isi_float = []
for i in isi:
    s = ""
    for j in i:
        if j!=" ":
            s+=j
    isi_float.append(float(s))
data['DC'] = np.array(isi_float)
data

data_headers= []
for x in data:
    data_headers.append(x.strip())
data.columns = data_headers
data_fwi = data['FWI'].to_numpy()
data_fwi_clean = [] 
for i in data_fwi:
    try:
        data_fwi_clean.append(float(i))
    except:
        print(i)
data_fwi_median = np.median(data_fwi_clean)
for i in range(len(data_fwi)):
    try:
        float(data_fwi[i])
    except:
        data_fwi[i] = data_fwi_median
data['FWI'] = data_fwi

from sklearn.model_selection import train_test_split
X = data[data_headers]
y = data['Classes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40,random_state=16)
# X_train,y_train = X_train.reshape(-1,1),y_train.reshape(-1,1)
X_train,y_train
# y_train

from sklearn.linear_model import LogisticRegression
# logisticRegr = LogisticRegression(random_state=16,solver='lbfgs',max_iter=1000)
logReg = LogisticRegression(random_state=16,max_iter=1000)
logReg.fit(X_train, y_train)

y_pred = logReg.predict(X_test)
y_pred

from sklearn import metrics

conf_matrix = metrics.confusion_matrix(y_test, y_pred)
conf_matrix

"""## Dawing Roc Curve"""

y_pred_proba = logReg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

