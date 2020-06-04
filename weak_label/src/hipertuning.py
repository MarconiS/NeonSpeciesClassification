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
bsvc = BaggingClassifier(base_estimator=SVC(probability = True), 
                         n_jobs = 2, random_state=0,
                         n_estimators=10)
params = {
 'baggingclassifier__base_estimator__C': [500, 1000, 1500], #C = 1000
}

bsvc = GridSearchCV(estimator=make_pipeline(StandardScaler(), bsvc),  
                    param_grid=params, 
                    cv=3,
                    refit=True)
bsvc.fit(X_res, train_y)
best_bsvc = bsvc.best_params_

### hipertune RFC ####
mlp = MLPClassifier(random_state=0, learning_rate = 'adaptive')
params = {'mlpclassifier__solver': ['lbfgs', 'adam'], 
          #'mlpclassifier__max_iter': [4000], 
          'mlpclassifier__alpha': [10, 20,50,100], 
          'mlpclassifier__hidden_layer_sizes': [10,12,15], 
}

mlp_g = GridSearchCV(estimator=make_pipeline(StandardScaler(), mlp), 
                    param_grid=params, 
                    cv=3, 
                    refit=True)
mlp_g.fit(X_res, train_y)
best_mlp = mlp_g.best_params_

from sklearn.neighbors import KNeighborsClassifier
### hipertune KNN ####
knn = KNeighborsClassifier(n_jobs = 2)
params = {
 'kneighborsclassifier__weights': ["uniform", "distance"], 
 'kneighborsclassifier__n_neighbors': [1,3,7,15,20,30,50],
 }
knn_g = GridSearchCV(estimator=make_pipeline(StandardScaler(),knn), 
                    param_grid=params, 
                    cv=3,
                    refit=True)

knn_g.fit(X_res, train_y)
best_knn = knn_g.best_params_

from sklearn.naive_bayes import GaussianNB
### hipertune NB ####
mnb = make_pipeline(StandardScaler(),GaussianNB())
mnb.fit(X_res, train_y)
best_mnb = mnb_g.best_params_
