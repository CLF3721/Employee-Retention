# Google Advanced Data Analytics Certification

## Description

Capstone Project for the Google Advanced Data Analytics Certification course.

### Scenario:

* The Human Resources (HR) department of a large consulting firm, Salifort Motors, wants to take some initiatives to improve employee satisfaction levels at the company since it is so time-consuming and expensive to find, interview, and hire new employees.

### Problem Question:

* What is likely to make the employee leave the company?

### Task:

* Predict whether or not an employee will leave the company and provide data-driven suggestions for HR on how to increase employee retention.

### Solution:

* Analyze the data collected by the HR department, build predictive model to identify empoyees likely to quit, and identify factors that contribute to their leaving if possibile.
* Algorithms: Logistic Regression, Decision Tree, Random Forest, XGBoost, K-Means

### Deliverables:

1. Executive summary that you would present to external stakeholders as the data professional in Salifort Motors; includes model evaluation, interpretation, data visualizations, ethical considerations, and the resources you used to troubleshoot and find answers or solutions.
2. Completed python scripts.

### Dataset:

[Kaggle HR Dataset](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction): 14,999 rows (employees), 10 columns

| COL_NAME              | TYPE  | DESC                                                              |
| :-------------------- | :---- | :---------------------------------------------------------------- |
| satisfaction_level    | int64 | The employee’s self-reported satisfaction level [0-1]            |
| last_evaluation       | int64 | Score of employee's last performance review [0-1]                 |
| number_project        | int64 | Number of projects employee contributes to                        |
| average_monthly_hours | int64 | Average number of hours employee worked per month                 |
| time_spend_company    | int64 | How long the employee has been with the company (years)           |
| work_accident         | int64 | Whether or not the employee experienced an accident while at work |
| left                  | int64 | Whether or not the employee left the company                      |
| promotion_last_5years | int64 | Whether or not the employee was promoted in the last 5 years      |
| department            | str   | The employee's department                                         |
| salary                | str   | The employee's salary (low, medium, or high)                      |

## Installation

```bash
pip install ipykernel numpy pandas matplotlib seaborn scikit-learn
```

## Imports

```python
# Data manipulation
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

# Saving models
import pickle

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Modeling - LogReg
from sklearn.linear_model import LogisticRegression

# Data Modeling - DecTree
from sklearn.tree import DecisionTreeClassifier

# Data Modeling - RandFor
from sklearn.ensemble import RandomForestClassifier

# Data Modeling - XGBoost
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

# Data Modeling - K-means
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Metrics & others
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree
```

## Code of Conduct

[Code of Conduct](https://www.python.org/psf/conduct/)