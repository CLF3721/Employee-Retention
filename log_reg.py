#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
@File      :   log_reg.py
@Time      :   2023/11/11 15:17:23
@Author    :   CLF
@Version   :   1.0
@Contact   :   https://www.linkedin.com/in/clf3721
@License   :   (C)Copyright 2023, CLF under the MIT License
@Desc      :   Google Advanced Data Analytics Certification Capstone project.

"""

###> Required Modules & Packages
##-> Data manipulation
import numpy as np
import pandas as pd
import eda
import feature_eng as fe

##-> Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

##-> Modeling: Logistic Regression
from sklearn.linear_model import LogisticRegression

##-> Metrics and helpful functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


###> Datasets
##-> Load dataset from eda.py
df_logreg = eda.df_prepped.copy()
df_logreg.head()

##-> Load dataset from feature_eng.py
df_logreg_fe = fe.df_prepped_fe.copy()
df_logreg_fe.head()



###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>
###~> Binomial Logistic Regression #1
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>
##-> Target Variable
y = df_logreg['LEFT']

##-> Feature Selection
X = df_logreg.drop(columns='LEFT')

##-> Data Partitioning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

##-> Build, fit, test the model
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)
y_pred = log_clf.predict(X_test)

##-> Confusion matrix *Key at the bottom of this script
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=log_clf.classes_)
log_disp.plot(values_format='')
plt.show()

##-> Check class imbalance
df_logreg['LEFT'].value_counts(normalize=True) # approx 83%-17% split ==> doesn't need resampling

##-> Generate classification report
target_names = ['Predicted would not leave', 'Predicted would leave']
print(classification_report(y_test, y_pred, target_names=target_names))


'''
###> BLR Insights
The classification report shows that the logistic regression model achieved a precision of 79%, recall of 82%, f1-score of 80% (all weighted averages), and accuracy of 82%. However, if it's most important to predict employees who leave, then the scores are significantly lower.
'''




###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>
###~> Binomial Logistic Regression #2
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>
"""
The first model build round included all variables as features. 
Round #2 will incorporate the feature engineered df to build improved models. 
"""
##-> Repeated steps above with FE dataframe
y = df_logreg_fe['LEFT']
X = df_logreg_fe.drop('LEFT', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)
y_pred = log_clf.predict(X_test)

##-> Confusion matrix #2
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=log_clf.classes_)
log_disp.plot(values_format='')
plt.show()

##-> Generate a classification report #2
target_names = ['Predicted would not leave', 'Predicted would leave']
print(classification_report(y_test, y_pred, target_names=target_names))


'''
###> Feature Engineering and BLR #2 Insights
    Not much change after feature engineering/augmentation
'''





'''
Key: 

True negatives  |   False positives
-----------------------------------
False negatives |   True positives

True negatives: The number of people who did not leave that the model accurately predicted did not leave.
False positives: The number of people who did not leave the model inaccurately predicted as leaving.
False negatives: The number of people who left that the model inaccurately predicted did not leave
True positives: The number of people who left the model accurately predicted as leaving

*A perfect model would yield all true negatives and true positives, and no false negatives or false positives.
'''
