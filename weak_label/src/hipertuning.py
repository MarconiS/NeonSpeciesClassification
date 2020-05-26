#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:06:14 2020

@author: sergiomarconi
"""

import numpy as np
import pandas as pd

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier
from mlxtend.classifier import SoftmaxRegression
from sklearn.ensemble import GradientBoostingClassifier


from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

le = LabelEncoder()
le.fit(y_res.taxonID)
train_y = le.transform(y_res.taxonID)

### hipertune RFC ####
rf = RandomForestClassifier(random_state=0, n_jobs = 2, oob_score = True)
params = {
 'randomforestclassifier__n_estimators': [100,500,1000], #500
 'randomforestclassifier__max_features': ['sqrt', 'auto', 'None'], #sqrt
 'randomforestclassifier__criterion' : ['gini', 'entropy'], #entropy
 }
rf_g = GridSearchCV(estimator=make_pipeline(StandardScaler(), rf), 
                    param_grid=params, 
                    cv=3, 
                    refit=True)
rf_g.fit(X_res, train_y)
best_rg = rf_g.best_params_
import pickle 
htune_pt = "./weak_label/hipertune/"
file_pi = open(htune_pt+'prf.obj', 'w') 
pickle.dump(rf_g, file_pi)

### hipertune Hist GB ####
gb = HistGradientBoostingClassifier(random_state=0)
params = {
 'histgradientboostingclassifier__max_iter': [100,500,1000], #1000
 'histgradientboostingclassifier__learning_rate': [0.1], #0.1
 'histgradientboostingclassifier__max_depth': [25, 75], #25
 'histgradientboostingclassifier__loss': ['categorical_crossentropy'],
 'histgradientboostingclassifier__l2_regularization': [0,0.5,1.5], #0.5
 }
gb = GridSearchCV(estimator=make_pipeline(StandardScaler(), gb), 
                    param_grid=params, 
                    cv=3,
                    refit=True)
gb.fit(X_res, train_y)
best_gb = gb.best_params_



### hipertune BC with SVC ####
bsvc = BaggingClassifier(base_estimator=SVC(probability = True), n_jobs = 2, oob_score = True, random_state=0)
params = {
 'baggingclassifier__base_estimator__C': [5, 1.0, 0.1, 0.01],
}

bsvc = GridSearchCV(estimator=make_pipeline(StandardScaler(), bsvc),  
                    param_grid=params, 
                    cv=3,
                    refit=True)
bsvc.fit(X_res, train_y)
best_bsvc = bsvc.best_params_
