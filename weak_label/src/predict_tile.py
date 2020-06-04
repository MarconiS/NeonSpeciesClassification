#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:39:48 2020

@author: sergiomarconi
"""
#load model
mod = joblib.load('./weak_label/mods/upperTrees_final_model.pkl')
#make predictions
tile = pd.read_csv("./weak_label/pred_indir/725000_46960002_image.csv")
did = np.unique(tile.domainID)
tile.domainID = domain_encode[did[0]]
sid = np.unique(tile.siteID)
tile.siteID = site_encode[sid[0]]
individualID = tile["individualID"]
tile_x = tile.drop(columns=["individualID"])
predict_tile = mod.predict_proba(tile_x)
predict_tile = pd.DataFrame(predict_tile)
eval_an = pd.concat([individualID, predict_tile], axis=1)
eval_an = eval_an.groupby(['individualID'], as_index=False).mean()

# get the column name of max values in every row
taxa_classes
eval_an.columns =  np.append(['individualID'], taxa_classes)
preds = eval_an.drop(columns=['individualID'])
pred_itc = preds.idxmax(axis=1)

pred_itc = pd.concat([eval_an.individualID, pred_itc], axis=1)
#save prediction
pred_itc.to_csv("./weak_label/pred_out/taxa_725000_46960002_image.csv")
