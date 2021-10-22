# -*- coding: utf-8 -*-
"""
Modified Sept 2021

@author: Miguel V. Martin for ML course (based on Scikit-learn)
"""

import pandas as pd
sal_data = pd.read_csv('attributes_vs_salary.dat')

# Create training/testing datasets
from sklearn.model_selection import train_test_split
train, test = train_test_split(sal_data, test_size=0.2)#, random_state=42)
train_labels = train['Income ($K/year)'] >=100 #labels are Boolean
test_labels = test['Income ($K/year)'] >=100 #labels are Boolean
train_data = train.iloc[:,1:3] #take only YofEd and age columns
test_data = test.iloc[:,1:3]


# Graph <100 and >=100 with different shapes
import matplotlib.pyplot as plt
over100_yoe = test.loc[test['Income ($K/year)'] >= 100].iloc[:,1]
under100_yoe = test.loc[test['Income ($K/year)'] < 100].iloc[:,1]
over100_age = test.loc[test['Income ($K/year)'] >= 100].iloc[:,2]
under100_age = test.loc[test['Income ($K/year)'] < 100].iloc[:,2]
plt.scatter(over100_yoe, over100_age, color='g', label='over $100K', marker='^', s = 70)
plt.scatter(under100_yoe, under100_age, color='b', label='under $100K', marker='v', s = 70)
plt.legend()
plt.xlabel("years of education")
plt.ylabel("age")

# Fit a Gaussian Naive Bayes classifier and see its performance
from sklearn.naive_bayes import GaussianNB
import numpy as np
gnb = GaussianNB()
pred = gnb.fit(train_data, train_labels).predict(test_data)
#print("Number of mislabeled points out of a total %d points: %d. Accuracy: %f" 
#     % (len(train_data),(train_labels != pred).sum(), 1-(train_labels != pred).sum()/len(train_data)))
print("Number of mislabeled points out of a total %d points: %d. Accuracy: %f" 
     % (len(test_data),(test_labels != pred).sum(), 1-(test_labels != pred).sum()/len(test_data)))

from sklearn.metrics import confusion_matrix
print('Confusion matrix (TN,FP/FN,TP):\n', confusion_matrix(test_labels, pred))
from sklearn.metrics import precision_score, recall_score
print('Precision:', precision_score(test_labels, pred))
print('Recall:', recall_score(test_labels, pred))
from sklearn.metrics import f1_score
print('F1 Score:', f1_score(test_labels, pred))

# Graph contour of our classification model 
x0s = np.linspace(-1, 31, 100)
x1s = np.linspace(0, 60, 100)
x0, x1 = np.meshgrid(x0s, x1s)
X = np.c_[x0.ravel(), x1.ravel()]
y_pred = gnb.predict(X).reshape(x0.shape)
plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
plt.show()
