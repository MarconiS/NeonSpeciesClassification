#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:39:48 2020

@author: sergiomarconi
"""
import joblib
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

site_to_domanin = {
        'GUAN':"D04",
        'ABBY':"D16",
        'BART':"D01",
        "BONA":'D19',
        "CLBJ":"D11",
        'DEJU':"D19",
        'DELA':"D08",
        'DSNY':"D03",
        'GRSM' :"D07",
        'HARV' :"D01",
        'HEAL':"D19",
        'LENO':"D08",
        'MLBS' : "D07",
        'NIWO': "D13",
        'OSBS':'D03',
        'SCBI' : "D02",
        'SERC':'D02',
        'SOAP':"D17",
        'STEI': "D05",
        'TEAK':"D17",
        'UKFS': "D06",
        'KONZ': "D06",
        'MOAB': "D13",
        'ONAQ': "D15",
        'RMNP':"D10",
        'SRER': "D14",
        'TALL': "D08",
        'YELL': "D12",
        }
 


def get_tile_boxes_ids(pt, ras_pt):
    import geopandas as gpd
    import rasterio
    from rasterio import features
    boxes = gpd.read_file(pt)
    boxes['individualID']=boxes.index
    rst = rasterio.open(ras_pt)
    meta = rst.meta.copy()
    rst.close()
    #meta.update(compress='lzw')
    meta.update(dtype='int32')
    with rasterio.open("tmp", 'w+', **meta) as out:
        out_arr = out.read(1)  
        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = ((geom,value) for geom, value in zip(boxes.geometry, boxes.individualID))
        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        #out.write_band(1, burned)
#    
    dim_br = burned.shape
    burned = np.reshape(burned, (dim_br[0]*dim_br[1], 1))
    return(burned)
   
    

    
    
def tile_surface_elevation(full_path):
    import h5py
    hdf5_file = h5py.File(full_path, 'r')
    file_attrs_string = str(list(hdf5_file.items()))
    file_attrs_string_split = file_attrs_string.split("'")
    sitename = file_attrs_string_split[1]
    Smooth_Surface_Elevation = hdf5_file[sitename]["Reflectance/Metadata/Ancillary_Imagery/Smooth_Surface_Elevation"].value
    dim_dem = Smooth_Surface_Elevation.shape
    Smooth_Surface_Elevation = np.reshape(Smooth_Surface_Elevation, (dim_dem[0]*dim_dem[1], 1))
    return(Smooth_Surface_Elevation)





#pt = "/Volumes/Stele/brdf_tiles/GRSM_270000_3951000_.tif"
def preprocess_tile(pt):
    import rasterio
    import glob
    brick = rasterio.open(pt)
    brick = brick.read()
    brick = np.swapaxes(brick,0,2)
    brick = np.swapaxes(brick,0,1)
    br_shape = brick.shape
    brick = np.reshape(brick, (br_shape[0]*br_shape[1], br_shape[2]))
    max_pix = np.apply_along_axis(max, 1, brick)
    min_pix = np.apply_along_axis(min, 1, brick)
    #filter for greennes and shadows
    ndvi = (brick[:,89]- brick[:,57])/(brick[:,57] + brick[:,89]) >0.5
    nir860 = (brick[:,95] + brick[:,96])/20000 >0.2
    mask = (max_pix < 10000) * (min_pix > -1)  * ndvi * nir860
    brick = brick[np.squeeze(mask)]
    brick = brick[:,18:357]
    # normMat = np.apply_along_axis(sum, 1, np.square(brick))
    # normMat = np.sqrt(normMat)
    # normMat =  np.tile(normMat, (1, 350))
    brick = normalize(brick)
    #normMat = matrix(data=rep(normMat,ncol(brick)),ncol=ncol(brick))
    #brick=brick/normMat
    #now apply the knd reduction
    kld_groups = pd.read_csv("./dimensionality_reduction/kld_30_grps.csv")
    all_data = np.zeros([brick.shape[0],1])
    for jj in kld_groups._kld_grp.unique():
        which_bands = kld_groups._kld_grp == jj
        #min
        new_col = np.apply_along_axis(min, 1, brick[:,which_bands])[...,None]
        all_data = np.append(all_data, new_col, 1)
        #mean
        new_col = np.apply_along_axis(np.mean, 1, brick[:,which_bands])[...,None]
        all_data = np.append(all_data, new_col, 1)
        #max
        new_col = np.apply_along_axis(max, 1, brick[:,which_bands])[...,None]
        all_data = np.append(all_data, new_col, 1)
#
    #
    #add metadata: site and domain
    tileID = pt.split("/")[-1]
    siteID= tileID[9:13]
    domainID = site_to_domanin[siteID]
    ele_tile = tileID[0:-4]
    ele_tile = glob.glob('/orange/ewhite/NeonData/**/*'+ele_tile+'.h5', recursive = True)
    elevation = tile_surface_elevation(ele_tile[0])
    elevation = elevation[np.squeeze(mask)]
    #add metadata:elevation
    all_data = pd.DataFrame(all_data[:,1:90])
    all_data.insert(0, 'siteID', siteID)
    all_data.insert(1, 'domainID', domainID)
    all_data.insert(2, 'elevation', elevation)
    #add metadata: boxes ids
    box_tile = tileID.split("_")
    box_tile = box_tile[2]+"*"+box_tile[4]+"_"+box_tile[5]+"_image.shp"
    box_tile = glob.glob('/orange/idtrees-collab/predictions/*'+box_tile, recursive = False)
    if len(box_tile) is 0:
        deepBoxes = np.array(range(0, 1000000))
    else:
        deepBoxes = get_tile_boxes_ids(box_tile[0], pt)
#
    deepBoxes = deepBoxes[np.squeeze(mask)]
    all_data.insert(0, 'individualID', deepBoxes)
    return(all_data)

  
    



def preprocess_df(df):
    meta_col = df.filter(items=['individualID', 'taxonID', 'siteID', 'domainID', 'elevation', 'stemDiameter', 'height', 'growthForm', 'plantStatus', 'canopyPosition', 'nlcdClass', 'Easting', 'Northing', 'itcLongitude', 'itcLatitude'])
    df = df.filter(like="band")
    df = df.iloc[:,18:357]
    #filter for greennes and shadows
    ndvi = (df.iloc[:,89]- df.iloc[:,57])/(df.iloc[:,57] + df.iloc[:,89]) >0.5
    nir860 = (df.iloc[:,95] + df.iloc[:,96])/2 >0.2
    df = normalize(df)
    #now apply the knd reduction
    kld_groups = pd.read_csv("./dimensionality_reduction/kld_30_grps.csv")
    all_data = np.zeros([df.shape[0],1])
    for jj in kld_groups._kld_grp.unique():
        which_bands = kld_groups._kld_grp == jj
        #min
        new_col = np.apply_along_axis(min, 1, df[:,which_bands])[...,None]
        all_data = np.append(all_data, new_col, 1)
        #mean
        new_col = np.apply_along_axis(np.mean, 1, df[:,which_bands])[...,None]
        all_data = np.append(all_data, new_col, 1)
        #max
        new_col = np.apply_along_axis(max, 1, df[:,which_bands])[...,None]
        all_data = np.append(all_data, new_col, 1)
    #add metadata
    all_data = pd.DataFrame(all_data[:,1:90])
    all_data = pd.concat([meta_col, all_data], axis=1)
    all_data.to_csv("./weak_label/kld_30_classes.csv", index=False)
    
    

    
  # df = pd.read_csv("./weak_label/indir/csv/dataset_marconi2020_brdf_centers.csv")  
    
#tl_nm = "/orange/idtrees-collab/hsi_brdf_corrected/corrHSI/NEON_D13_NIWO_DP3_448000_4430000_reflectance.tif"  
#load model
tl_nm = sys.argv[1]
mod = joblib.load('./weak_label/mods/final80_model.pkl')
domain_encode = pd.read_csv('./weak_label/mods/domain_encode__final80.csv')
site_encode = pd.read_csv('./weak_label/mods/site_encode__final80.csv')
taxa_classes = pd.read_csv('./weak_label/mods/taxonID_dict__final80.csv')
outdir = "/orange/ewhite/s.marconi/crownMaps/"
#make predictions
tile = preprocess_tile(tl_nm) #"./pred_indir/"+"HARV_732000_4713000__brdf_itc.csv")
tileID = tl_nm.split("/")[-1]
tile.to_csv(outdir+'/raw_csv/'+tileID[0:-4]+"_df.csv")

did = np.unique(tile.domainID)
did = domain_encode[domain_encode['domainID']==did[0]]
tile.domainID = did.domainE.iloc[0]
sid = np.unique(tile.siteID)
sid = site_encode[site_encode['siteID']==sid[0]]
tile['siteID'] = sid.siteE.iloc[0]
individualID = tile["individualID"]
tile = tile.drop(columns=["individualID"])
predict_tile = mod.predict_proba(tile)
predict_tile = pd.DataFrame(predict_tile)
del(tile)
predict_tile = pd.concat([individualID, predict_tile], axis=1)
#save prediction
predict_tile.to_csv(outdir+'/tile_preds_csv/'+tileID[0:-4]+"tile_prediction.csv", index=False)

#append to deepForest predictions
eval_an = predict_tile.groupby(['individualID'], as_index=False).mean()
del(predict_tile)
# get the column name of max values in every row
taxa_classes
eval_an.columns =  np.append(['individualID'], taxa_classes.iloc[:,1])
preds = eval_an.drop(columns=['individualID'])
probmax = np.apply_along_axis(max, 1, preds)
pred_itc = preds.idxmax(axis=1)
pred_itc = pd.concat([eval_an['individualID'], pred_itc, pd.Series(probmax)], axis=1)

#link to geopandas
import geopandas as gpd
import glob
#add metadata: boxes ids
box_tile = tileID.split("_")
box_tile = box_tile[2]+"*"+box_tile[4]+"_"+box_tile[5]+"_image.shp"
box_tile = glob.glob('/orange/idtrees-collab/predictions/*'+box_tile, recursive = False)
pred_itc = pred_itc.rename(columns={0: 'taxonID', 1: 'probability'})
#if len(box_tile) is 0:
pred_itc.to_csv(outdir+'/predictions_csv/'+tileID[0:-4]+"_itc_prediction.csv", index=False)
#else:
boxes = gpd.read_file(box_tile[0])
boxes['individualID']=boxes.index
boxes = boxes.merge(pred_itc, on='individualID')
boxes.to_csv(outdir+'/predictions_csv/'+tileID[0:-4]+"_itc_prediction.csv", index=False)

