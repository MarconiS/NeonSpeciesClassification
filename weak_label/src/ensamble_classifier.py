#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:12:17 2020

@author: sergiomarconi
"""

from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import StratifiedShuffleSplit
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

from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier

from sklearn.svm import SVC


# #resample classes
# smote_tomek = SMOTETomek(random_state=0)

# #define models
# rf = RandomForestClassifier(bRF,random_state=0)

# mlp = MLPClassifier(solver='lbfgs',random_state=0, alpha=mlp_alpha, 
#                                      hidden_layer_sizes=mlp_ls)
# gb = GradientBoostingClassifier(pgb)

# svm = SVC(svm_param)

# #define ensamble
# estimators = [('rf', rf), ('svr', svm),  ('mlp', mlp), ('gb', gb)]
# clf = StackingClassifier(estimators=estimators, 
#         final_estimator=LogisticRegressionCV(cv=3, random_state=0),n_jobs=3)
# clf.fit(X_resampled, y_resampled).score(X_test, y_test)



#ensamble of class imbalanced models
from imblearn.ensemble import BalancedRandomForestClassifier
brf = BalancedRandomForestClassifier(n_estimators = 100, random_state=0, oob_score=True)
from imblearn.ensemble import RUSBoostClassifier
rusboost = RUSBoostClassifier(learning_rate = 0.01, n_estimators = 100,
                             algorithm='SAMME.R', random_state=0)
from sklearn.ensemble import BaggingClassifier
bc_svm = BaggingClassifier(base_estimator=SVC(),  oob_score = True,  random_state=0)



#define ensamble
bestimators = [('rf', brf),  
               ('gb', rusboost), 
               ('svm',make_pipeline(StandardScaler(),BaggingClassifier(n_estimators = 300)))] #('bSVM', bc), ('bMLPC', bc_mlpc)
clf_imbl = StackingClassifier(classifiers = [make_pipeline(StandardScaler(),brf), 
                                             make_pipeline(StandardScaler(),rusboost), 
                                             make_pipeline(StandardScaler(),BaggingClassifier())],
                              use_probas=True,
                          average_probas=False,
                          meta_classifier=LogisticRegressionCV())
#clf_imbl.get_params().keys()
#rf_check = brf.fit(X_res, y_res.taxonID)

#hipertune imbalanced models
#params = {'meta_classifier__Cs': [0.1, 10.0]}

from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(estimator=clf_imbl, param_grid=params, 
#                     cv=5,
#                     refit=True)
# #calibrate model
# grid.fit(X_res, y_res.taxonID)
#train model
clf_imbl.fit(X_res, y_res.taxonID)
rf = RandomForestClassifier(random_state=0, oob_score = True, n_jobs = 2)
rf.fit(X_res, y_res.taxonID)
clf_imbl.score(X_test, y_test['taxonID'])
#make prediction on test_set
predict_test = clf_imbl.predict_proba(X_test)
predict_test = pd.DataFrame(predict_test)



from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  classification_report
from sklearn.metrics import multilabel_confusion_matrix

# format outputs for final evaluation
taxa_classes = rf.classes_
colnames = np.append(['individualID', 'taxonID'], taxa_classes) #.tolist()
y_test.reset_index(drop=True, inplace=True)
predict_test.reset_index(drop=True, inplace=True)
eval_df = pd.concat([y_test, predict_test], axis=1)
eval_df.columns = colnames
#aggregate probabilities of each pixel to return a crown based evaluation. Using majority vote
eval_df = eval_df.groupby(['individualID', 'taxonID'], as_index=False).mean()

y_itc = eval_df['taxonID']
pi_itc = eval_df.drop(columns=['individualID', 'taxonID'])

# get the column name of max values in every row
pred_itc = pi_itc.idxmax(axis=1)
cm = confusion_matrix(y_itc, pred_itc, labels = taxa_classes)
cm = pd.DataFrame(cm, columns = taxa_classes, index = taxa_classes)
#mcm = multilabel_confusion_matrix(y_itc, pred_itc)
#f1_score(y_itc, pred_itc, average='macro')
#f1_score(y_itc, pred_itc, average='micro')
report = classification_report(y_itc, pred_itc, output_dict=True)
report = pd.DataFrame(report).transpose()
report
eval_df.to_csv("./weak_predictions.csv")

final_report= {}
for i in ('y_itc', 'pi_itc', 'pred_itc','clf_imbl', 'cm', 'report'):
    final_report[i] = locals()[i]
with open('./imbl_report', 'wb') as f:
  pickle.dump(final_report, f, protocol=pickle.HIGHEST_PROTOCOL)

