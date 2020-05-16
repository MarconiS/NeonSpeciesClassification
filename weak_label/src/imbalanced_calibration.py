#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 23:41:55 2020

@author: sergiomarconi
"""
from sklearn.model_selection import learning_curve, GridSearchCV

hyperRF = {
    'n_estimators': [100, 300, 500, 800, 1500],
    'max_depth': [3,5, 8, 15, 25, 30],
    'min_samples_split': [2, 5, 10, 15, 100],
    'min_samples_leaf': [1, 2, 5, 10],
}
gridF = GridSearchCV(estimator =BalancedRandomForestClassifier(), 
        param_grid = hyperRF, cv = 3, verbose = 1, n_jobs = 2)
RF = gridF.fit(X_train, y_train)
pRF = gridF.best_params_


# tune gb
hyperGB = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 
    'n_estimators': [100, 300, 500, 800, 1500]}
    #'min_samples_split': [2, 5, 10, 15, 100]}
    #'max_depth': [3,5, 8, 15, 25, 30],
    #'min_samples_leaf': [1, 2, 5, 10]}

tuning = GridSearchCV(estimator =RUSBoostClassifier( algorithm='SAMME.R',
          random_state=10), 
          param_grid = hyperGB, scoring='accuracy',n_jobs=-2, cv=3)
tuning.fit(X_train, y_train)
pgb = tuning.best_params_

# tune mlpc
hyperBC = {
    'n_estimators': [100, 300, 500, 800, 1500],
    'max_samples': [1, 2, 5, 10, 15, 100],
    'max_features': [1,4,8,16,32]
    
}
mlpc = GridSearchCV(BaggingClassifier(base_estimator=SVC()), 
                    hyperBC, n_jobs=2, cv=3)
mlpc.fit(X_train, y_train)
pmlp = mlpc.best_params_

hyperNN = {
    'n_estimators': [100, 300, 500, 800, 1500],
  #  'max_samples': [1, 2, 5, 10, 15, 100],
#    'max_features': [1,4,8,16,32]
    
}
mlpc = GridSearchCV(BaggingClassifier(base_estimator=SVC()), 
                    hyperNN, n_jobs=2, cv=3)
mlpc.fit(X_train, y_train)
pmlp = mlpc.best_params_

