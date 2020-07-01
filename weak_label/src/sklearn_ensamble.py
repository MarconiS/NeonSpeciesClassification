
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:12:17 2020

@author: sergiomarconi
"""

import numpy as np
import pandas as pd

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegressionCV

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier

from sklearn.svm import SVC


#define models
rf = RandomForestClassifier(random_state=0, oob_score = True, n_jobs = 2)
#gb = GradientBoostingClassifier(random_state=0)
gb = HistGradientBoostingClassifier(random_state=0, n_jobs = 2, loss = 'categorical_crossentropy')
clf = BaggingClassifier(base_estimator=SVC(), n_jobs = 2, random_state=0)

# Initializing models
clf_bl = StackingClassifier(classifiers = [make_pipeline(StandardScaler(),rf), 
                                             make_pipeline(StandardScaler(),gb), 
                                             make_pipeline(StandardScaler(),clf],
                              use_probas=True,
                          average_probas=False,
                          meta_classifier=LogisticRegressionCV())

params = {'kneighborsclassifier__n_neighbors': [1, 5],
          'randomforestclassifier__n_estimators': [10, 50],
          'meta_classifier__C': [0.1, 10.0]}

grid = GridSearchCV(estimator=sclf, 
                    param_grid=params, 
                    cv=3,
                    refit=True)
grid.fit(X, y)
cv_keys = ('mean_test_score', 'std_test_score', 'params')


