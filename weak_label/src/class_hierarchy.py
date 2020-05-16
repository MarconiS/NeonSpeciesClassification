#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:56:26 2020

@author: sergiomarconi
"""
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import h_fbeta_score, multi_labeled

species_to_genus = {
        'ABBA':"AB",
        'ABLAL':"AB",
        'ACNE2': "AC",
        'ACPE':"AC",
        'ACRU': "AC",
        'ACSA3' : "AC",
        'ACSA2' :"AC",
        'AIAL' : "AI",
        'ALRU2': "AL",
        'AMLA' : "AM",
        'BEAL2': "BE",
        'BEPA': "BE",
        'BELE': "BE",
        'BENE4' : "BE",
        'BUBU':"BU",
        'BUSI':"BU",
        'CACA18':"CA1",
        'CADE27':"CA2",
        'CAGL8':"CA3",
        'CAOV2':"CA3",
        'CACO15': "CA3",
        'CATO6':"CA3",
        'CAIL2':"CA3",
        'CECA4':"CE1",
        'CELA':"CE2",
        'CEOC':"CE2",
        'CODR':"CO",
        'CODI8':"CO",
        'COFL2':"CO2",
        'ELAN':"EL",
        'FAGR':"FA",
        'FRAM2':"FR",
        'FRPE':"FR",
        'GYDI':"GY",
        'GUOF':"GU",
        'GLTR':"GL",
        'HALES':"HA",
        'JUNI':"JU1",
        'JUVI':"JU2",
        'LIST2':"LI1",
        'LITU':"LI2",
        'MAPO':"MA",
        'MORU2':"MO",
        'NYBI':"NY",
        'NYSY':"NY",
        'OXYDE':"OX",
        'PICEA':"PI1",
        'PIAL3':"PI2",
        'PICOL':"PI2",
        'PIEL':"PI2",
        'PIEN':"PI2",
        'PIFL2':"PI2",
        'PIGL':"PI2", 
        'PIMA':"PI2",
        'PIPA2':"PI2",
        'PIPO':"PI2", 
        'PIRU':"PI2",
        'PIST':"PI2",
        'PITA':"PI2",
        'PINUS':"PI2",
        'PLOC':"PL",
        'POTR5':"PO",
        'POGR4':"PO",
        'PODE3':"PO",
        'PRVE':"PR",
        'PSME':"PS",
        'QUAL':"QU", 
        'QUCO2':"QU",
        'QUERC':"QU",
        'QUGE2':"QU", 
        'QULA3':"QU",
        'QULY':"QU", 
        'QUMA3':"QU",
        'QUMI':"QU",
        'QUMO4':"QU",
        'QUMU':"QU",
        'QUNI':"QU",
        'QUPA5':"QU",
        'QURU':"QU",
        'QUST':"QU",
        'RHGL':"RH",
        'SASSA':'SA',
        'SYOC':"SY",
        'TRSE6':"TR",
        'TSCA':"TS",
        'TIAM':"TI",
        'ULAL':"UL",
        'ULAM':"UL", 
        'ULCR':"UL",
        'ULRU':"UL",
        }



#class heirarchy
class_hierarchy = {
        ROOT: ["AB", "AC", "AM", "BE", "BU", "CA1", "CA2",
               "CA3", "CE1", "CE2", "CO", "EL", "FA", "FR", "HA", 
               "JU1", "JU2", "LI1", "LI2", "MA", "NY", "OX", "PI1",
               "PI2", "PL", "PO", "PR", "QU", "RH", "SY", "TR", "TS", "UL"],
        "AB": ['ABBA', 'ABLAL'],
        "AC": ['ACNE2', 'ACRU', 'ACSA3'],
        "AM": ['AMLA'],
        "BE": ['BEAL2', 'BELE','BENE4'],
        "BU": ['BUBU', 'BUSI'],
        "CA1": ['CACA18'],
        "CA2": ['CADE27'],
        "CA3": ['CAGL8', 'CAOV2','CATO6'],
        "CE1": ['CECA4'],
        "CE2": ['CELA', 'CEOC'],
        "CO": ['CODR'],
        "EL": ['ELAN'],
        "FA": ['FAGR'],
        "FR": ['FRAM2','FRPE'],
        "HA": ['HALES'],
        "JU1": ['JUNI'],
        "JU2": ['JUVI'],
        "LI1": ['LIST2'],
        "LI2": ['LITU'],
        "MA": ['MAPO'],
        "NY": ['NYBI','NYSY'],
        "OX": ['OXYDE'],
        "PI1": ['PICEA'],
        "PI2": ['PIAL3','PICOL', 'PIEL', 'PIEN', 'PIFL2', 'PIGL', 'PIMA', 'PIPA2', 'PIPO', 'PIRU', 'PIST','PITA', 'PINUS'],
        "PL": ['PLOC'],
        "PO": ['POTR5'],
        "PR": ['PRVE'],
        "QU": ['QUAL', 'QUCO2', 'QUERC', 'QUGE2', 'QULY', 'QUMA3', 'QUMI', 'QUMO4', 'QUMU', 'QUNI', 'QUPA5', 'QURU','QUST'],
        "RH": ['RHGL'],
        "SY": ['SYOC'],
        "TR": ['TRSE6'],
        "TS": ['TSCA'],
        "UL": ['ULAL', 'ULAM', 'ULCR','ULRU'],
    }

def launch_hierarchical_modeling_pipeline():
    # Used for seeding random state
    RANDOM_STATE = 42
    clf_himbl = StackingClassifier(classifiers = [make_pipeline(StandardScaler(with_mean=False),brf), 
                                                 make_pipeline(StandardScaler(with_mean=False),rusboost), 
                                                 make_pipeline(StandardScaler(with_mean=False),BaggingClassifier())],
                                  use_probas=True,
                              average_probas=False,
                              meta_classifier=LogisticRegressionCV())
    
    
    hclf_brf = HierarchicalClassifier(
            base_estimator=make_pipeline(StandardScaler(with_mean=False),brf),
            class_hierarchy=class_hierarchy,
    )
    hclf_rusboost = HierarchicalClassifier(
            base_estimator=make_pipeline(StandardScaler(with_mean=False),rusboost),
            class_hierarchy=class_hierarchy,
    )
    hclf_bbgsvm = HierarchicalClassifier(
            base_estimator=make_pipeline(StandardScaler(with_mean=False),BaggingClassifier()),
            class_hierarchy=class_hierarchy,
    )
    clf_himbl = StackingClassifier(classifiers = [hclf_brf, hclf_rusboost, hclf_bbgsvm],
                                    use_probas=True,
                              average_probas=False,
                              meta_classifier=LogisticRegressionCV())
                                   
    clf_himbl.fit(X_res, y_res['taxonID'] )
    
    clf_himbl.score(X_test, y_test['taxonID'])
    #make prediction on test_set
    predict_test = clf_himbl.predict_proba(X_test)
    predict_test = pd.DataFrame(predict_test)
    
    
    
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import  classification_report
    from sklearn.metrics import multilabel_confusion_matrix
    # format outputs for final evaluation
    hclf_brf.fit(X_res, y_res['taxonID'] )
    taxa_classes = hclf_brf.classes_
    colnames = np.append(['individualID', 'taxonID'], taxa_classes[33:104]) #.tolist()
    y_test.reset_index(drop=True, inplace=True)
    predict_test.reset_index(drop=True, inplace=True)
    eval_df = pd.concat([y_test, predict_test], axis=1)
    eval_df.columns = colnames
    #aggregate probabilities of each pixel to return a crown based evaluation. Using majority vote
    eval_df = eval_df.groupby(['individualID', 'taxonID'], as_index=False).mean()
    
    y_itc = eval_df['taxonID']
    pi_itc = eval_df.drop(columns=['individualID', 'taxonID'])
    sp_itc = eval_df.iloc[:,35:104]
    gen_itc = eval_df.iloc[:,2:35]
    # get the column name of max values in every row
    pred_itc = sp_itc.idxmax(axis=1)
    gen_itc = gen_itc.idxmax(axis=1)
    cm = confusion_matrix(y_itc, pred_itc)
    cm = pd.DataFrame(cm, columns = taxa_classes, index = taxa_classes)
    #mcm = multilabel_confusion_matrix(y_itc, pred_itc)
    #f1_score(y_itc, pred_itc, average='macro')
    #f1_score(y_itc, pred_itc, average='micro')
    report = classification_report(y_itc, pred_itc, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report
    
    final_report= {}
    for i in ('y_itc', 'pi_itc', 'pred_itc','clf_himbl', 'cm', 'report'):
        final_report[i] = locals()[i]
    with open('./imbl_report_hmod', 'wb') as f:
      pickle.dump(final_report, f, protocol=pickle.HIGHEST_PROTOCOL)
    
