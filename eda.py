#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
@File      :   eda.py
@Time      :   2023/11/11 15:17:23
@Author    :   CLF
@Version   :   1.0
@Contact   :   https://www.linkedin.com/in/clf3721
@License   :   (C)Copyright 2023, CLF under the MIT License
@Desc      :   Google Advanced Data Analytics Certification Capstone project.

#Env setup
select interp
create venv
activate venv: .venv\Scripts\Activate.ps1

installs:
pip install --upgrade pip setuptools wheel build configparser pipreqs
pip install ipykernel numpy pandas matplotlib seaborn scikit-learn
pip-compile .venv\pyvenv.cfg requirements.txt
cat requirements.txt
pip freeze > requirements.txt

"""

###> Required Modules & Packages
##-> Data manipulation
import numpy as np
import pandas as pd

##->! Workspace Configuration
import warnings; warnings.filterwarnings("ignore")
import sys; print("User Current Version:-", sys.version)
import time; startTime = time.time()
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.colheader_justify", "left")
pd.set_option("display.max_info_columns", 300000)
pd.set_option("display.max_info_rows", 300000)

##-> Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

##-> Saving models
import pickle



###~~~~~~~~~~~~~~~~~~>
###~> pAce - ANALYZE
###~~~~~~~~~~~~~~~~~~>
##-> Load dataset
df = pd.read_csv('data\HR_capstone_dataset.csv')
df.head()
df.info() #no missing values
df.describe()

##-> Fix colnames
df.columns = [col.upper() for col in df.columns]
df = df.rename(columns={'NUMBER_PROJECT': 'PROJ_NUM','AVERAGE_MONTLY_HOURS': 'AVG_HRS_PER_MONTH', 'TIME_SPEND_COMPANY': 'TENURE', 'LAST_EVALUATION': 'LAST_EVAL', 'SATISFACTION_LEVEL': 'SATISFACTION'})
df.columns

##-> Drop duplicates
df.duplicated().sum()
df[df.duplicated()].head()
df_drop_dup = df.drop_duplicates(keep='first', ignore_index=True)
df_drop_dup.info()


###> Outlier Investigation
##-> 'TENURE' boxplot
plt.figure(figsize=(6,6))
plt.title('Boxplot to detect outliers for tenure', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df_drop_dup['TENURE'])
plt.show()

##-> 25th percentile
percentile25 = df_drop_dup['TENURE'].quantile(0.25)

##-> 75th percentile
percentile75 = df_drop_dup['TENURE'].quantile(0.75)

##-> Interquartile
iqr = percentile75 - percentile25

##-> Upper/Lower limits
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

##-> 'TENURE' outliers
outliers = df_drop_dup[(df_drop_dup['TENURE'] > upper_limit) | (df_drop_dup['TENURE'] < lower_limit)]
# df_rmo = df_drop_dup[(df_drop_dup['TENURE'] >= lower_limit) & (df_drop_dup['TENURE'] <= upper_limit)]



###> EDA: Comparing employees who stayed vs left the company
##-> How many employees left?
df_drop_dup['LEFT'].value_counts()
df_drop_dup['LEFT'].value_counts(normalize=True)

##-> Boxplot 'AVG_HRS_PER_MONTH' vs 'PROJ_NUM'
fig, ax = plt.subplots(1, 2, figsize = (22,8))
sns.boxplot(data=df_drop_dup, x='AVG_HRS_PER_MONTH', y='PROJ_NUM', hue='LEFT', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Monthly hours by number of projects', fontsize='14')

##-> Histogram 'PROJ_NUM'
tenure_stay = df_drop_dup[df_drop_dup['LEFT']==0]['PROJ_NUM']
tenure_left = df_drop_dup[df_drop_dup['LEFT']==1]['PROJ_NUM']
sns.histplot(data=df_drop_dup, x='PROJ_NUM', hue='LEFT', multiple='dodge', shrink=2, ax=ax[1])
ax[1].set_title('Number of projects histogram', fontsize='14')
plt.show()

##-> Show employees with 7 projects (all left)
df_drop_dup[df_drop_dup['PROJ_NUM']==7]['LEFT'].value_counts()


##-> Scatterplot 'AVG_HRS_PER_MONTH' vs 'SATISFACTION'
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df_drop_dup, x='AVG_HRS_PER_MONTH', y='SATISFACTION', hue='LEFT', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'LEFT', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14')


##-> Boxplot 'SATISFACTION' vs 'TENURE'
fig, ax = plt.subplots(1, 2, figsize = (22,8))
sns.boxplot(data=df_drop_dup, x='SATISFACTION', y='TENURE', hue='LEFT', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Satisfaction by tenure', fontsize='14')

##-> Histogram 'TENURE'
tenure_stay = df_drop_dup[df_drop_dup['LEFT']==0]['TENURE']
tenure_left = df_drop_dup[df_drop_dup['LEFT']==1]['TENURE']
sns.histplot(data=df_drop_dup, x='TENURE', hue='LEFT', multiple='dodge', shrink=5, ax=ax[1])
ax[1].set_title('Tenure histogram', fontsize='14')
plt.show()


##-> Satisfaction scores
df_drop_dup.groupby(['LEFT'])['SATISFACTION'].agg([np.mean,np.median])


##-> Histogram 'SALARY' vs 'TENURE' (short vs long tenured employees)
fig, ax = plt.subplots(1, 2, figsize = (22,8))
tenure_short = df_drop_dup[df_drop_dup['TENURE'] < 7]
tenure_long = df_drop_dup[df_drop_dup['TENURE'] >= 7]

sns.histplot(data=tenure_short, x='TENURE', hue='SALARY', discrete=1, 
             hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.5, ax=ax[0])
ax[0].set_title('Salary histogram by tenure: short-tenured people', fontsize='14')

sns.histplot(data=tenure_long, x='TENURE', hue='SALARY', discrete=1, 
             hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.4, ax=ax[1])
ax[1].set_title('Salary histogram by tenure: long-tenured people', fontsize='14')


##-> Scatterplot 'AVG_HRS_PER_MONTH' vs 'LAST_EVAL'
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df_drop_dup, x='AVG_HRS_PER_MONTH', y='LAST_EVAL', hue='LEFT', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'LEFT', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14')


##-> Scatterplot 'AVG_HRS_PER_MONTH' vs 'PROMOTION_LAST_5YEARS'
plt.figure(figsize=(16, 3))
sns.scatterplot(data=df_drop_dup, x='AVG_HRS_PER_MONTH', y='PROMOTION_LAST_5YEARS', hue='LEFT', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'LEFT', 'stayed'])
plt.title('Monthly hours by promotion last 5 years', fontsize='14')


##-> Display counts for each department
df_drop_dup["DEPARTMENT"].value_counts()

##-> Stacked Histogram 'DEPARTMENT' vs 'LEFT'
plt.figure(figsize=(11,8))
sns.histplot(data=df_drop_dup, x='DEPARTMENT', hue='LEFT', discrete=1, hue_order=[0, 1], multiple='dodge', shrink=.5)
plt.xticks(rotation=45)
plt.title('Counts of stayed/left by department', fontsize=14)


##-> Encode 'SALARY' as an ordinal numeric category
df_enc = df_drop_dup.copy()
df_enc['SALARY'] = (
    df_enc['SALARY'].astype('category')
    .cat.set_categories(['low', 'medium', 'high'])
    .cat.codes
)

##-> Drop 'DEPARTMENT' since it didn't much impact departed employees
df_numerical = df_enc.drop(columns='DEPARTMENT')

##-> Get dummies for 'DEPARTMENT' data for modeling
df_enc = pd.get_dummies(df_enc, drop_first=False)


##-> Correlation heatmap
plt.figure(figsize=(16, 9))
heatmap = sns.heatmap(df_numerical.corr(), vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12)


'''
###* EDA Insights
It appears that employees are leaving the company as a result of poor management. 
    Leaving is tied to longer working hours, many projects, and generally lower satisfaction levels. *All employees with 7 projects left the company. 
    It can be ungratifying to work long hours and not receive promotions or good evaluation scores. There's a sizeable group of employees at this company who are probably burned out. 
    It also appears that if an employee has spent more than six years at the company, they tend not to leave. 

'''


###> Performance Metrics
print("Execution time in seconds: " + str(round((time.time() - startTime), 2)))
