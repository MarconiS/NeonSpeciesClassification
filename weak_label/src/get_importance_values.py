#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:27:47 2020

@author: sergiomarconi
"""


from yellowbrick.model_selection import FeatureImportances


from mlxtend.evaluate import feature_importance_permutation
imp_vals, all_val = feature_importance_permutation(
    predict_method=clf_bl.predict, 
    X=np.array(X_test),
    y=np.array(y_test.taxonID.ravel()),
    metric='accuracy',
    num_rounds=1,
    seed=1)


imp_vals.shape
all_val.shape
pd.Series(imp_vals).to_csv('./weak_label/mods'+siteID+'_model_importance_vals.csv', index = False)
pd.DataFrame(all_val).to_csv('./weak_label/mods'+siteID+'_model_all_vals.csv', index = False)
