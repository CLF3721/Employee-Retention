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

##-> Data visualization
import matplotlib.pyplot as plt

##-> Modeling: Logistic Regression
from sklearn.linear_model import LogisticRegression

##-> Metrics and helpful functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

##-> Load dataset from eda.py
df_logreg = eda.df_enc_no_ouliers.copy()
df_logreg.head()



###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>
###~> paCe - CONSTRUCT: Binomial Logistic Regression 
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>
#* BLR suits the task because it involves binary classification.
##-> Target Variable - what we want the model to predict
y = df_logreg['LEFT']
y.head()

##-> Feature Selection
X = df_logreg.drop(columns='LEFT')
X.head()

# Split the data into training set and testing set. Don't forget to stratify based on the values in 'y', since the classes are imbalanced.
##-> Data Partitioning: Split the data into training, testing, and hold out samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

##-> Build the model and fit it to the training dataset
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)

##-> Test the model by using it to make predictions on the test set
y_pred = log_clf.predict(X_test)

##-> Confusion matrix
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=log_clf.classes_)
log_disp.plot(values_format='')
plt.show()

# Create a classification report that includes precision, recall, f1-score, and accuracy metrics to evaluate the performance of the logistic regression model.
# Check the class balance in the data. In other words, check the value counts in the 'left' column. Since this is a binary classification task, the class balance informs the way you interpret accuracy metrics. If the data is severely imbalanced, you might want to resample the data to make it more balanced. In this case, you can use this data without modifying the class balance and continue evaluating the model.

##-> Check class imbalance
df_logreg['LEFT'].value_counts(normalize=True)
# approx 83%-17% split ==> doesn't need resampling

##-> Generate a classification report
target_names = ['Predicted would not leave', 'Predicted would leave']
print(classification_report(y_test, y_pred, target_names=target_names))



'''
###> BLR Insights
The classification report shows that the logistic regression model achieved a precision of 79%, recall of 82%, f1-score of 80% (all weighted averages), and accuracy of 82%. However, if it's most important to predict employees who leave, then the scores are significantly lower.
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
