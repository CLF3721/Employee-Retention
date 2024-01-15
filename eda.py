#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
@File      :   eda.py
@Time      :   2023/11/11 15:17:23
@Author    :   CLF
@Version   :   1.0
@Contact   :   https://www.linkedin.com/in/clf3721
@License   :   (C)Copyright 2023, CLF under the MIT License
@Desc      :   Exploratory Data Analysis (EDA) for the Google Advanced Data Analytics Certification Capstone project.

installs:
pip install ipykernel pycaret sweetviz

"""


###~~~~~~~~~~~>
###~> Imports
###~~~~~~~~~~~>
##-> Data manipulation
import numpy as np
import pandas as pd

##-> Workspace Config
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

##-> Data visualization
import matplotlib.pyplot as plt
import sweetviz as sv
import seaborn as sns



###~~~~~~~~~~~~~~~~~~~~~~~~>
###~> Data Familiarization
###~~~~~~~~~~~~~~~~~~~~~~~~>
##-> Load dataset
df = pd.read_csv('data\HR_capstone_dataset.csv')
# df.head()
# df.info() #no missing values
# df.describe()

##-> Fix colnames
df.columns = [col.upper() for col in df.columns]
df = df.rename(columns={'NUMBER_PROJECT': 'PROJ_NUM','AVERAGE_MONTLY_HOURS': 'AVG_HRS_PER_MONTH', 'TIME_SPEND_COMPANY': 'TENURE', 'LAST_EVALUATION': 'LAST_EVAL', 'SATISFACTION_LEVEL': 'SATISFACTION'})
df = df[['LEFT', 'PROJ_NUM', 'AVG_HRS_PER_MONTH', 'TENURE', 'PROMOTION_LAST_5YEARS', 'SATISFACTION', 'LAST_EVAL', 'WORK_ACCIDENT', 'SALARY', 'DEPARTMENT']]

##-> Drop duplicates
df.duplicated().sum()
# df[df.duplicated()].head()
df_drop_dup = df.drop_duplicates(keep='first', ignore_index=True)
# df_drop_dup.info()

##-> Outlier Investigation: 'TENURE'
# plt.figure(figsize=(6,6))
# plt.title('Boxplot to detect outliers for tenure', fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# sns.boxplot(x=df_drop_dup['TENURE'])
# outliers_tenure_plt = plt.show()

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



###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>
###~> Exploratory Data Analysis (EDA)
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>
# ##-> Compare employees who stayed vs left the company
# df_drop_dup['LEFT'].value_counts()
# df_drop_dup['LEFT'].value_counts(normalize=True)

# ##-> Boxplot 'AVG_HRS_PER_MONTH' vs 'PROJ_NUM' + Histogram 'PROJ_NUM'
# fig, ax = plt.subplots(1, 2, figsize = (22,8))
# sns.boxplot(data=df_drop_dup, x='AVG_HRS_PER_MONTH', y='PROJ_NUM', hue='LEFT', orient="h", ax=ax[0])
# ax[0].invert_yaxis()
# ax[0].set_title('Monthly hours by number of projects', fontsize='14')
# tenure_stay = df_drop_dup[df_drop_dup['LEFT']==0]['PROJ_NUM']
# tenure_left = df_drop_dup[df_drop_dup['LEFT']==1]['PROJ_NUM']
# sns.histplot(data=df_drop_dup, x='PROJ_NUM', hue='LEFT', multiple='dodge', shrink=2, ax=ax[1])
# ax[1].set_title('Number of projects histogram', fontsize='14')
# plt.show()

# ##-> Show employees with 7 projects (all left)
# df_drop_dup[df_drop_dup['PROJ_NUM']==7]['LEFT'].value_counts()

# ##-> Scatterplot 'AVG_HRS_PER_MONTH' vs 'SATISFACTION'
# plt.figure(figsize=(16, 9))
# sns.scatterplot(data=df_drop_dup, x='AVG_HRS_PER_MONTH', y='SATISFACTION', hue='LEFT', alpha=0.4)
# plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
# plt.legend(labels=['166.67 hrs./mo.', 'LEFT', 'stayed'])
# plt.title('Monthly hours by last evaluation score', fontsize='14')

# ##-> Boxplot 'SATISFACTION' vs 'TENURE' + Histogram 'TENURE'
# fig, ax = plt.subplots(1, 2, figsize = (22,8))
# sns.boxplot(data=df_drop_dup, x='SATISFACTION', y='TENURE', hue='LEFT', orient="h", ax=ax[0])
# ax[0].invert_yaxis()
# ax[0].set_title('Satisfaction by tenure', fontsize='14')
# sns.histplot(data=df_drop_dup, x='TENURE', hue='LEFT', multiple='dodge', shrink=5, ax=ax[1])
# ax[1].set_title('Tenure histogram', fontsize='14')
# plt.show()

# ##-> Satisfaction scores
# df_drop_dup.groupby(['LEFT'])['SATISFACTION'].agg([np.mean,np.median])

# ##-> Histogram 'SALARY' vs 'TENURE' (short vs long tenured employees)
# fig, ax = plt.subplots(1, 2, figsize = (22,8))
# tenure_short = df_drop_dup[df_drop_dup['TENURE'] < 7]
# tenure_long = df_drop_dup[df_drop_dup['TENURE'] >= 7]
# sns.histplot(data=tenure_short, x='TENURE', hue='SALARY', discrete=1, 
#              hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.5, ax=ax[0])
# ax[0].set_title('Salary histogram by tenure: short-tenured people', fontsize='14')
# sns.histplot(data=tenure_long, x='TENURE', hue='SALARY', discrete=1, 
#              hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.4, ax=ax[1])
# ax[1].set_title('Salary histogram by tenure: long-tenured people', fontsize='14')

# ##-> Scatterplot 'AVG_HRS_PER_MONTH' vs 'LAST_EVAL'
# plt.figure(figsize=(16, 9))
# sns.scatterplot(data=df_drop_dup, x='AVG_HRS_PER_MONTH', y='LAST_EVAL', hue='LEFT', alpha=0.4)
# plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
# plt.legend(labels=['166.67 hrs./mo.', 'LEFT', 'stayed'])
# plt.title('Monthly hours by last evaluation score', fontsize='14')

# ##-> Scatterplot 'AVG_HRS_PER_MONTH' vs 'PROMOTION_LAST_5YEARS'
# plt.figure(figsize=(16, 3))
# sns.scatterplot(data=df_drop_dup, x='AVG_HRS_PER_MONTH', y='PROMOTION_LAST_5YEARS', hue='LEFT', alpha=0.4)
# plt.axvline(x=166.67, color='#ff6361', ls='--')
# plt.legend(labels=['166.67 hrs./mo.', 'LEFT', 'stayed'])
# plt.title('Monthly hours by promotion last 5 years', fontsize='14')

# ##-> Display counts for each department
# df_drop_dup["DEPARTMENT"].value_counts()

# ##-> Stacked Histogram 'DEPARTMENT' vs 'LEFT'
# plt.figure(figsize=(11,8))
# sns.histplot(data=df_drop_dup, x='DEPARTMENT', hue='LEFT', discrete=1, hue_order=[0, 1], multiple='dodge', shrink=.5)
# plt.xticks(rotation=45)
# plt.title('Counts of stayed/left by department', fontsize=14)

# ##-> Stacked bar chart 'LEFT' vs 'DEPARTMENT'
# pd.crosstab(df_drop_dup['DEPARTMENT'], df_drop_dup['LEFT']).plot(kind ='bar',color='mr')
# plt.title('Counts of employees who left versus stayed across department')
# plt.ylabel('Employee count')
# plt.xlabel('DEPARTMENT')
# plt.show()

# ##-> Promotions
# promo5 = df_drop_dup[df_drop_dup['PROMOTION_LAST_5YEARS']==1]
# promo5.describe(include='all')
# promo5[promo5['LEFT']==1].sort_values(by='PROJ_NUM')

# ##-> work accident
# wk_accd = df_drop_dup[df_drop_dup['WORK_ACCIDENT']==1]#1850
# wk_accd.describe(include='all')
# wk_accd[wk_accd['LEFT']==1].sort_values(by='DEPARTMENT').describe(include='all')#105
# #! This could be why 105 employes left out of the 1991 total - maybe remove from sample?

# ##-> salary
# pay = df_drop_dup[df_drop_dup['SALARY']=='high']
# pay.describe(include='all')
# pay[pay['PROMOTION_LAST_5YEARS']==1].sort_values(by='DEPARTMENT')
# pay[pay['LEFT']==1].sort_values(by='PROJ_NUM').describe()

# ##-> dept
# dept = df_drop_dup[df_drop_dup['DEPARTMENT']=='management']
# dept.info(verbose=True)
# dept.describe(include='all')
# dept[dept['LEFT']==1].sort_values(by='SALARY')



###~~~~~~~~~~~~~~~~~~~>
###~> Data Enrichment
###~~~~~~~~~~~~~~~~~~~>
df_enriched = df_drop_dup.copy()

# ##-> Inspect min/max average monthly hours values
# print('Min hours:', df_enriched['AVG_HRS_PER_MONTH'].min())
# print('Max hours:', df_enriched['AVG_HRS_PER_MONTH'].max())

##-> 'OVERWORKED' = working > 175 hrs/month and they have more than 3 projects.
df_enriched['OVERWORKED'] = 0
df_enriched['OVERWORKED'] = (((df_enriched['AVG_HRS_PER_MONTH']>175) & (df_enriched['PROJ_NUM']>3)) | (df_enriched['PROJ_NUM']>4)).astype("int64")
df_enriched['OVERWORKED'].value_counts()

##-> 'UNDERAPPRECIATED' = tenure>3yrs with no promotion or just overworked
df_enriched['UNDERAPPRECIATED'] = 0
df_enriched['UNDERAPPRECIATED'] = (((df_enriched['PROMOTION_LAST_5YEARS']==0) & (df_enriched['TENURE']>2)) | ((df_enriched['PROMOTION_LAST_5YEARS']==1) & (df_enriched['OVERWORKED']==1) & (df_enriched['SALARY']=='low'))).astype("int64")
# df_enriched['UNDERAPPRECIATED'].value_counts()

##-> Lack of employee growth opportunities
# df_enriched.groupby(['LEFT','PROJ_NUM'])['PROMOTION_LAST_5YEARS'].mean()
# df_enriched.groupby(['LEFT','OVERWORKED','PROMOTION_LAST_5YEARS']).mean()
hmmm = df_enriched[(df_enriched['LEFT']==1) & (df_enriched['WORK_ACCIDENT']==0) & (df_enriched['OVERWORKED']==1) & (df_enriched['TENURE']>3) & (df_enriched['PROMOTION_LAST_5YEARS']==1)]
lets_check = df_enriched[(df_enriched['LEFT']==1) & (df_enriched['WORK_ACCIDENT']==0) & (df_enriched['OVERWORKED']==1) & (df_enriched['UNDERAPPRECIATED']==1)]
# df_enriched.head()
# Hardly anyone surveyed was given a promotion.  Many of the duplicates were those marked as 1.  Of those that were promoted in the past 5 years that left, regardless of tenure and promotion, they were still being underpaid.

##-> df with just tgt var and new cols
df_binary = df_enriched[['LEFT', 'OVERWORKED', 'UNDERAPPRECIATED']].sort_values(by=['LEFT','OVERWORKED', 'UNDERAPPRECIATED'], ignore_index=True)

##-> Calculating proportions
grouped = df_binary.groupby('LEFT').mean()
# proportions = grouped.div(grouped.sum(), axis=0)
# # Plotting
# proportions.plot(kind='bar', stacked=True)
# plt.title('Proportion of Employees Leaving by Overworked and Underappreciated Status')
# plt.ylabel('Proportion')
# plt.legend(title='Left', loc='upper left')
# plt.show()



###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>
###~> Encoding Categorical Values
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>
##-> 'SALARY' is categorical but since it's ordinal and a hierarchy to the categories, it's better not to dummy this column, but rather to encode 'SALARY' as an ordinal numeric category, 0-2.
df_enc = df_drop_dup.copy()
df_enc['SALARY'] = (df_enc['SALARY'].astype('category').cat.set_categories(['low', 'medium', 'high']).cat.codes).astype("int64")

##-> Get dummies for 'DEPARTMENT' categoricals
df_enc_dummied = pd.get_dummies(df_enc, drop_first=False)

# ##-> Correlation heatmap
# plt.figure(figsize=(16, 9))
# heatmap = sns.heatmap(df_enc_dummied.corr(), vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
# heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12)

##-> Drop 'DEPARTMENT' since it didn't much impact departed employees for tighter correlation
df_enc.drop(columns=['DEPARTMENT'], inplace=True)
# df_enc.groupby(['LEFT']).mean()

# ##-> Tighter Correlation heatmap
# plt.figure(figsize=(16, 9))
# heatmap = sns.heatmap(df_enc.corr(), vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
# heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12)

# ##-> Drop outliers if using alg sensitive to them like LogRegs
# df_enc_no_ouliers = df_enc[~df_enc['TENURE'].isin(outliers['TENURE'])]
# grpd_no_ouliers = df_enc_no_ouliers.groupby(['LEFT', 'PROJ_NUM'])['SATISFACTION', 'AVG_HRS_PER_MONTH', 'TENURE', 'PROMOTION_LAST_5YEARS'].mean()



###~~~~~~~~~~~~~~~~~~>
###~> EDA - SweetViz
###~~~~~~~~~~~~~~~~~~>
# sv.config_parser.read("Override.ini")
sv_report = sv.analyze(df_enc_dummied, target_feat="LEFT")
sv_report.show_html("Salifort-Motors-Employee-Retention-Report.html", open_browser=False, layout='vertical')



'''
###> EDA Insights
It appears that employees are leaving the company as a result of poor management. 
Leaving is tied to longer working hours, many projects, and generally lower satisfaction levels. *All employees with 7 projects left the company. 
It can be ungratifying to work long hours and not receive promotions or good evaluation scores. There's a sizeable group of employees at this company who are probably burned out. 
It also appears that if an employee has spent more than six years at the company, they tend not to leave. 

'''
