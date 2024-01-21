#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

'''
@File      :   pycaret_model.py
@Time      :   2024/01/15 01:26:11
@Author    :   CLF
@Version   :   1.0
@Contact   :   https://www.linkedin.com/in/clf3721
@License   :   (C)Copyright 2024, CLF under the MIT License
@Desc      :   Model COnstruction with PyCaret

'''


###~~~~~~~~~~~>
###~> Imports
###~~~~~~~~~~~>
##-> Get prepared dataset from eda.py
import eda

##-> Model Construction with PyCaret
# import pycaret
# pycaret.__version__
from pycaret.classification import *



###~~~~~~~~~~~~~~~~~~~~~~~~~>
###~> Data Load & Partition
###~~~~~~~~~~~~~~~~~~~~~~~~~>
##-> Load dataset
data = eda.df_enc.copy()
# data.info()

##-> Chop the data up into training sample and unseen sample for testing
df = data.sample(frac=0.90, random_state=37)
df.reset_index(drop=True, inplace=True)
df_unseen = data.drop(df.index)
df_unseen.reset_index(drop=True, inplace=True)



###~~~~~~~~~~~~~~~~~~~~~~>
###~> Model Construction
###~~~~~~~~~~~~~~~~~~~~~~>
##-> PyCaret Initial SetUp
experiment = setup(df, target='LEFT',
                   train_size=0.8,
                   transformation=True,
                   normalize=True,
                   profile=True,
                   session_id=42)
# get_config()

##-> Compare baseline models & Blend
top2 = experiment.compare_models(n_select=2, sort='F1')

# ##-> Create the model, train it, then fine tune hyperparameters
# top2_blended = blend_models(top2)
# tuned_blended_model = tune_model(top2_blended, choose_better=True)

###> Not using blended bc can't plot metrics
##-> Create the model, train it, then fine tune hyperparameters
lgtbst = experiment.create_model('lightgbm')
tuned_best_model = tune_model(lgtbst, choose_better=True)
xt = get_config('X_test')
yt = get_config('y_test')

##-> Model Evaluation
eval_model = evaluate_model(tuned_best_model)

##-> Finalize the model for deployment
final_model = finalize_model(tuned_best_model)

##-> Plot model performance metrics
plot_model(final_model, plot='confusion_matrix', save=True)
plot_model(final_model, plot='class_report', save=True)
plot_model(final_model, plot='learning', save=True)
plot_model(final_model, plot='feature', save=True)



###~~~~~~~~~~~~~~~~~~~~~>
###~> Model Validaation
###~~~~~~~~~~~~~~~~~~~~~>
##-> Validation: Pred on df with final model
pred_vald = predict_model(final_model)
# pred_vald.head()



###~~~~~~~~~~~~~~~~~>
###~> Model Testing
###~~~~~~~~~~~~~~~~~>
##-> Test: Pred on df_unseen
pred_test = predict_model(final_model, data=df_unseen)
# pred_test.head()



###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>
###~> Model Saved for Deployment
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>
##-> Save the model
save_model(final_model, 'final_model/lgtbst_model')


'''
For each feature observation (dots), the red color represents high values, while the blue color represents low values.

If the dots are more spread to the right, this means that the feature has a high effect on the positive class (Attrition = 1). Similarly, if the dots are spread to the left, the feature has a negative effect on attrition.

### Looking at the graph, we can conclude that:
##### Features that have a great effect on turnover
* Low monthly income
* Low total working years
* Low age
* High overtime hours
##### Features that have a medium effect on turnover
* Low years at the company
* A high number of companies worked before
* High distance from home
* High years since last promotion

'''




