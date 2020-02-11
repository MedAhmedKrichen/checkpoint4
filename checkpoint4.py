import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from matplotlib import pyplot as plt

df=pd.read_csv("train_and_test2.csv")
print(df.head())

y=df["2urvived"].values
X=df["Passengerid"].values[:,np.newaxis]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=30)
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
"""
"""


confusion_matrix=pd.crosstab(y_test,y_pred)
print(confusion_matrix)

"""
ROC curve stands for receiver operating characteristic curve is the curve resulting of changing thresholds lines
in False Positive Rate (FPR) for the 0 axis and True Positive Rate (TPR) for the 1 axis.(in some cases we replace the TPR by precision rate)

AUC stand for Area Under the ROC Curve is basicly the area of a ROC curve and it's used for comparing ROC curves  
"""
log_ROC_auc=metrics.roc_auc_score(y_test,y_pred)
fpr,trp,thresholds=metrics.roc_curve(y_test,y_pred)

plt.figure()
plt.plot(fpr,trp)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()





