#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:22:28 2020

@author: sergiomarconi
"""
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from keras.layers import Embedding
from keras.layers.merge import concatenate
from keras.utils import plot_model
from keras.layers import Input
from keras.layers.merge import concatenate
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import ADASYN
from collections import Counter

# prepare input data
def prepare_inputs(X_train, X_test, cats = ['domainID', 'siteID']):
	X_train_enc, X_test_enc = list(), list()
	# label encode each column
	for i in  cats:
		le = LabelEncoder()
		le.fit(X_train[i])
		# encode
		train_enc = le.transform(X_train[i])
		test_enc = le.transform(X_test[i])
		# store
		X_train_enc.append(train_enc)
		X_test_enc.append(test_enc)
	return X_train_enc, X_test_enc


# prepare target
def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc


def categorical_encoder(cats,y, tr_lb):
    import category_encoders as ce
    le = LabelEncoder()
    le.fit(y[tr_lb])
    le = le.transform(y[tr_lb])
    enc = ce.LeaveOneOutEncoder(cols=['domainID', 'siteID'])
    # enc = enc.fit(cats).transform(cats)
    train_enc = enc.fit_transform(cats[tr_lb],le)
    return(train_enc)
    
siteID =  None #"D01"
dim_red = None #"pca"
max_threshold = 350
too_rare = False


#data = pd.read_csv("./weak_label/indir/csv/brdf_june5Ttop_hist.csv")

data = pd.read_csv("./weak_label/test_plus_30_classes.csv") 
#data = pd.read_csv("./centers_august_30k.csv") 
data = data.drop(['growthForm', 'stemDiameter','plantStatus', 'canopyPosition', 'nlcdClass','height', 'Easting', 'Northing','itcLongitude','itcLatitude'], axis=1)
#data = data.drop(['elevation'], axis=1)
#data = pd.read_csv("./weak_label/indir/csv/bf2_top_reflectance.csv")

#data = data.drop(columns=['species', 'genus', 'genus_id'])
#data = pd.read_csv("/Users/sergiomarconi/Documents/Data/NEON/VST/vst_top_bf1_reflectance.csv")
is_bad_genus = ["ABIES", "BETUL", "FRAXI", "MAGNO", "SALIX", "2PLANT",
                "OXYDE", "HALES", "PINUS", "QUERC", "PICEA"]
is_bad_genus =  data['taxonID'].isin(is_bad_genus)


if dim_red is 'pca':
    #apply dimensionality reduction on data
    pca = PCA(n_components = 40)
    X = data.drop(columns=['individualID', 'taxonID','siteID','domainID', 'height', 'area','elevatn'])
    X = pd.DataFrame(X)
    attr = data[['individualID', 'taxonID','siteID','domainID', 'height', 'area','elevatn']]
    data = pd.concat([attr, X], axis=1)
if dim_red is 'kld':
    import sys
    sys.path.append("../hsi_toolkit_py")
    from dimensionality_reduction import hdr
    X = data.drop(columns=['individualID', 'taxonID','siteID','domainID', 'height', 'area','elevatn'])
    X = hdr.dimReduction(X, numBands = 20)
    X = pd.DataFrame(X)
    attr = data[['individualID', 'taxonID','siteID','domainID', 'height', 'area','elevatn']]
    data = pd.concat([attr, X], axis=1)
if too_rare is True:
    too_rare =  ["ACFA","ACPE","AMEL","ARVIM","BEAL2","BEPA","BOSU2",
                 "CAAQ2","CACO15","CAOV3","CODI8","DIVI5","GUOF",
                 "LELE10","MAFR","NYAQ2","OSVI","PIAC",
                 "PIVI2","POGR4","PRAV","PRSE2","QUFA","QULA2","QULA3",
                 "QUPH","QUSH","ROPS","SAAL5","SILA20", "TAHE","TIAM"] 
    is_too_rare =  data['taxonID'].isin(too_rare)
    data = data[~is_too_rare]

#filter by site
if siteID is not None:
    is_site =  data['domainID']==siteID
    data = data[is_site]

species_id = data.taxonID.unique()
# #divide X and Y
# X = data.drop(columns=['individualID', 'taxonID'])

#splin into train and test by chosing columns
train_ids = data[['individualID','siteID', 'taxonID']].drop_duplicates()
train_ids = train_ids.groupby(['siteID', 'taxonID'], 
                  group_keys=False).apply(lambda x: x.sample(int(len(x)/ 2)+1,
                                          random_state=1))
#embedd/encode categorical data
train_ids.to_csv("./weak_label/indir/train_ids.csv")
# split train test from individualIDs
train = data['individualID'].isin(train_ids['individualID'])
test = data[~data["individualID"].isin(train_ids['individualID'])]
X_train = data.drop(columns=['individualID', 'taxonID'])[train]
X_test = data.drop(columns=['individualID', 'taxonID'])[~train]
y_train =  data[['taxonID']][train]
y_test =  data[['individualID','taxonID']][~train]
y_test.to_csv("./weak_label/indir/y_test.csv")

y = data[['taxonID']]

#encode categorical values using an LOO-Encoder and associating only 1 value to each
cats = data[['domainID', 'siteID']]      
cat_encoder = categorical_encoder(cats,y['taxonID'], train)
cat_encoder = pd.DataFrame(np.hstack([cats[train],cat_encoder])).drop_duplicates()
cat_encoder.columns = ['domainID','siteID', 'domainE', 'siteE']

cat_encoder = cat_encoder.assign(domainE=pd.to_numeric(cat_encoder['domainE'], errors='coerce')) 
cat_encoder = cat_encoder.assign(siteE=pd.to_numeric(cat_encoder['siteE'], errors='coerce')) 
site_encode = cat_encoder.groupby('siteID')['siteE'].mean()
domain_encode = cat_encoder.groupby('domainID')['domainE'].mean()                  


# oversample using SMOTENC in order not to loose the categorical effects
#get relatie frequency of each class
ratios_for_each = Counter(y_train.taxonID)
ratios_for_each = pd.DataFrame.from_dict(ratios_for_each, orient='index').reset_index()
ratios_for_each.iloc[:,1] = ratios_for_each.iloc[:,1]

#max_threshold max limit is the most frequent tree
max_threshold = min(max(ratios_for_each.iloc[:,1])-1, max_threshold)
#get classes with less than max, and oversample them
thres_species = ratios_for_each.iloc[:,1] > max_threshold
thres_species = ratios_for_each[thres_species].iloc[:,0]
thres_species = y_train.taxonID.isin(thres_species)
x_tmp = X_train[thres_species]
y_tmp = y_train[thres_species]

#undersample  
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NeighbourhoodCleaningRule
    
if len(np.unique(y_tmp)) > 1:
    #rus = ClusterCentroids(random_state=0)
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(x_tmp.to_numpy(), y_tmp.to_numpy())
    X_resampled = pd.DataFrame(X_resampled)
    y_resampled = pd.DataFrame(y_resampled)
else:
    import random
    rindx = random.sample(range(0, len(y_tmp)), max_threshold)
    X_resampled = x_tmp.iloc[rindx,:]
    y_resampled = y_tmp.iloc[rindx,:]
    X_resampled = pd.DataFrame(X_resampled)
    y_resampled = pd.DataFrame(y_resampled)

#get classes with less than max, and oversample them
thres_species = ratios_for_each.iloc[:,1] <= max_threshold
thres_species = ratios_for_each[thres_species].iloc[:,0]
thres_species = y_train.taxonID.isin(thres_species)
x_tmp = X_train[thres_species]
y_tmp = y_train[thres_species]

# add undersampled portion of dataset and oversample
x_tmp.reset_index(drop=True, inplace=True)
X_resampled.reset_index(drop=True, inplace=True)
X_resampled = X_resampled.rename(columns= pd.Series(x_tmp.columns))
x_tmp = pd.concat([x_tmp, X_resampled], axis=0) 

y_tmp.reset_index(drop=True, inplace=True)
y_resampled.reset_index(drop=True, inplace=True)
y_resampled = y_resampled.rename(columns= pd.Series(y_tmp.columns))
y_tmp = pd.concat([y_tmp, y_resampled], axis=0) 

min_class = Counter(y_tmp.taxonID)
min_class = min_class[min(min_class, key=min_class.get)]
min_class = min(8, min_class)
#oversampling using SMOTE-ENCODER
#smote_adasym = SMOTENC(random_state=0, categorical_features = [90,91], k_neighbors = min_class-1)
smote_adasym = SMOTENC(random_state=0, categorical_features = [0,1], k_neighbors = min_class-1)
X_res, y_res = smote_adasym.fit_resample(x_tmp.to_numpy(), y_tmp.to_numpy().ravel())

X_res  = pd.DataFrame(X_res, columns = pd.Series(x_tmp.columns))
y_res  = pd.DataFrame(y_res, columns = pd.Series(y_tmp.columns))

#convert training and testing categorical columns into corresponding value
dom_float = np.zeros(X_res.shape[0])
site_float = np.zeros(X_res.shape[0])
for ii in range(X_res.shape[0]):
    foo = X_res.domainID.iloc[ii] == domain_encode.index
    dom_float[ii] = domain_encode[foo]
    foo = X_res.siteID.iloc[ii] == site_encode.index
    site_float[ii] = site_encode[foo]
    
X_res['domainID'] = dom_float
X_res['siteID'] = site_float

#convert categorical data in test set
dom_float = np.zeros(X_test.shape[0])
site_float = np.zeros(X_test.shape[0])

for ii in range(X_test.shape[0]):
    foo = X_test.domainID.iloc[ii] == domain_encode.index
    dom_float[ii] = domain_encode[foo]
    foo = X_test.siteID.iloc[ii] == site_encode.index
    site_float[ii] = site_encode[foo]

X_test['domainID'] = dom_float
X_test['siteID'] = site_float


# simple check that no test data is in the pool of training crowns
tr_nm = data.individualID[train]
tr_nm = np.unique(tr_nm)
tst_nm = y_test.individualID
tst_nm = np.unique(tst_nm)
check = np.isin(tst_nm, tr_nm)
print(sum(check))
check = X_res.columns

