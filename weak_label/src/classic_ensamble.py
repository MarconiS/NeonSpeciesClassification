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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier
from mlxtend.classifier import SoftmaxRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier

from sklearn.svm import SVC


#define models
rf = RandomForestClassifier(random_state=0, oob_score = True, n_jobs = 1, 
                            n_estimators = 500, max_features = 'sqrt', criterion = 'entropy')

knn = KNeighborsClassifier(n_jobs = 1, weights = 'uniform', n_neighbors = 1, p=2)
gb = HistGradientBoostingClassifier(random_state=0, max_iter = 1000, learning_rate = 0.1, 
                max_depth = 25, loss = 'categorical_crossentropy', l2_regularization = 0.5)
bsvc = BaggingClassifier(base_estimator=SVC(probability = True, C = 1000), n_jobs = 1, random_state=0)


# StackingCVClassifier models
clf_bl = StackingCVClassifier(classifiers = [make_pipeline(StandardScaler(),rf),  #StackingCVClassifier
                                             make_pipeline(StandardScaler(),gb),
                                             make_pipeline(StandardScaler(),bsvc),
                                             make_pipeline(StandardScaler(),knn)],
                          use_probas=True,
                          #average_probas=False,
                          meta_classifier= LogisticRegressionCV(max_iter =10000, Cs = 5))

# params = {
#  'meta_classifier__Cs': [0.1, 5, 10]
#  }

# grid = GridSearchCV(estimator=clf_bl, 
#                     param_grid=params, 
#                     cv=3,
#                     refit=True)
# grid.fit(X_res, y_res.taxonID.ravel())

clf_bl.fit(X_res, y_res.taxonID.ravel())
print(clf_bl.score(X_test, y_test['taxonID'].ravel()))

#rf_check = knn.fit(X_res, y_res.taxonID)
#rf_check.score(X_test, y_test['taxonID'].ravel())
# #hipertune imbalanced models
# params = {'kneighborsclassifier__n_neighbors': [1, 5],
#           'randomforestclassifier__n_estimators': [10, 50],
#           'meta_classifier__C': [0.1, 10.0]}

# grid = GridSearchCV(estimator=sclf, 
#                     param_grid=params, 
#                     cv=3,
#                     refit=True)
# grid.fit(X, y)
# cv_keys = ('mean_test_score', 'std_test_score', 'params')

predict_an = clf_bl.predict_proba(X_test)
predict_an = pd.DataFrame(predict_an)



from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from class_hierarchy import *
# format outputs for final evaluation
rf = rf.fit(X_res, y_res.taxonID)
taxa_classes = rf.classes_
colnames = np.append(['individualID', 'taxonID'], taxa_classes) #.tolist()
y_test.reset_index(drop=True, inplace=True)
predict_an.reset_index(drop=True, inplace=True)
eval_an = pd.concat([y_test, predict_an], axis=1)
eval_an.columns = colnames
#aggregate probabilities of each pixel to return a crown based evaluation. Using majority vote
eval_an = eval_an.groupby(['individualID', 'taxonID'], as_index=False).mean()

y_itc = eval_an['taxonID']
pi_itc = eval_an.drop(columns=['individualID', 'taxonID'])

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
# domain_encode.to_csv("./domain_encode_"+"_"+siteID+".csv")
# site_encode.to_csv("./site_encode_"+"_"+siteID+".csv")
# pd.DataFrame(taxa_classes).to_csv("./taxonID_dict_"+"_"+siteID+".csv")

# eval_an.to_csv("./weak_an_"+"_"+siteID+"_"+"kld_probabilities.csv")
# pd.DataFrame(np.c_[eval_an[['individualID','taxonID']], pred_itc]).to_csv("./weak_an_"+"_"+siteID+"_"+"kld_pairs.csv")

# final_report= {}
# for i in ('y_itc', 'pi_itc', 'pred_itc','clf_bl', 'cm', 'report'):
#     final_report[i] = locals()[i]
# with open('./an_report_'+siteID, 'wb') as f:
#   pickle.dump(final_report, f, protocol=pickle.HIGHEST_PROTOCOL)

obs = np.zeros(len(pred_itc)).astype(str)
pred = np.zeros(len(pred_itc)).astype(str)
for ii in range(len(pred_itc)): 
    obs[ii] = species_to_genus[y_itc[ii]]
    pred[ii] = species_to_genus[pred_itc[ii]]
    
rep_fam = classification_report(obs, pred, output_dict=True)
rep_fam = pd.DataFrame(rep_fam).transpose()
rep_fam 
test_families = pd.concat([pd.Series(obs), pd.Series(pred)], axis=1) 
test_families = test_families.rename(columns={0: "observed", 1: "predicted"})
test_families.to_csv("./weak_an_"+"_"+siteID+"_"+"family_predictions.csv")   
cm_fam = confusion_matrix(obs, pred)
cm_fam = pd.DataFrame(cm_fam, columns = rep_fam.index[0:-3], index = rep_fam.index[0:-3])

eval_an.to_csv("./weak_an_"+"_"+siteID+"_"+"kld_probabilities.csv")
pd.DataFrame(np.c_[eval_an[['individualID','taxonID']], pred_itc]).to_csv("./weak_an_"+"_"+siteID+"_"+"kld_pairs.csv")
domain_encode.to_csv("./domain_encode_"+"_"+siteID+".csv")
site_encode.to_csv("./site_encode_"+"_"+siteID+".csv")
pd.DataFrame(taxa_classes).to_csv("./taxonID_dict_"+"_"+siteID+".csv")


import joblib
mod_out_pt = "./weak_label/mods/"+siteID+'_model.pkl'
joblib.dump(clf_bl, mod_out_pt)

  