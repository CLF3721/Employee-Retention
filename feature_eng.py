#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

'''
@File      :   feature_eng.py
@Time      :   2024/01/03 00:28:50
@Author    :   CLF
@Version   :   1.0
@Contact   :   https://www.linkedin.com/in/clf3721
@License   :   (C)Copyright 2024, CLF under the MIT License
@Desc      :   Google Advanced Data Analytics Certification Capstone project.

The high scores could be a result of possible data leakage in either 'SATISFACTION' or 'AVG_HRS_PER_MONTH' columns; the company likely didn't report satisfaction level for all of its employees, and assuming some employees have already decided to quit, or have already been identified by management as people to be fired, they may be working fewer hours. 

Create a new feature, 'OVERWORKED', and drop the two features possibly causing data leakage.

166.67 is approximately the average number of monthly hours for someone who works 50 weeks per year, 5 days per week, 8 hours per day.

Being overworked = working more than 175 hours per month on average.

Reassign the 'OVERWORKED' using a boolean mask to make binary.
    df_prepped_fe['OVERWORKED'] > 175 creates a series of booleans, consisting of 'True' for every value > 175 and 'False' for every values < 176
    .astype(int) converts all 'True' to 1 and all 'False' to 0

'''

###> Import eda.py
import eda
df_prepped_fe = eda.df_prepped.copy()


###> Feature Engineering
##-> Inspect min/max average monthly hours values
print('Min hours:', df_prepped_fe['AVG_HRS_PER_MONTH'].min())
print('Max hours:', df_prepped_fe['AVG_HRS_PER_MONTH'].max())

##-> 'OVERWORKED' = working > 175 hrs/month
df_prepped_fe['OVERWORKED'] = (df_prepped_fe['AVG_HRS_PER_MONTH'] > 175).astype(int)
df_prepped_fe['OVERWORKED'].head(20)

##-> Drop 'AVG_HRS_PER_MONTH' & 'SATISFACTION'
df_prepped_fe = df_prepped_fe.drop(['AVG_HRS_PER_MONTH','SATISFACTION'], axis=1)
df_prepped_fe.head()
