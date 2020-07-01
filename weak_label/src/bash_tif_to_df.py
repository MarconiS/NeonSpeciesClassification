#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:19:44 2020

@author: sergiomarconi
"""

from shapely.geometry import mapping
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio as rasterio
from rasterio.plot import plotting_extent
import geopandas as gpd
from pathlib import Path
from rasterio.mask import mask
import rasterstats as rs
    #
from os import listdir
import pandas as pd
from os.path import isfile, join


itc_pt = "//orange/idtrees-collab/predictions/"
brdf_pt = "//orange/idtrees-collab/hsi_brdf_corrected/corrHSI"
out_pt = "//orange/idtrees-collab/hsi_brdf_corrected/brdf_to_csv"
tile = sys.args[1]
    
itc_path = list(Path(itc_pt).rglob("*.[s][h][p]"))
itc_path = pd.Series(itc_path).astype('str') 
tile_id = itc_path.str.contains(tile[5:])
itc_path = itc_path[tile_id]
tile_id = itc_path.str.contains(tile[0:4])
itc_path = itc_path[tile_id]
if len(itc_path) > 0:
    itcs = gpd.read_file(itc_path.iloc[0])
    itcs = itcs.centroid
    # with rio.open(brdf_path) as brdf:
    #     brick = brdf.read()
    #create a 1m buffer around each center
    itc = itcs.buffer(1)
    itc = itc.envelope 
    geoms = itc.geometry.values # list of shapely geometries
    #itc['ID'] = range(itc.shape[0])
    # extract the geometry in GeoJSON format
    # transform to raster (quicker than loop and crop
    #get raster path
    brdf_path = list(Path(brdf_pt).rglob("*.[t][i][f]"))
    brdf_path = pd.Series(brdf_path).astype('str') 
    tile_id = brdf_path.str.contains(tile[5:])
    brdf_path = brdf_path[tile_id]
    tile_id = brdf_path.str.contains(tile[0:4])
    brdf_path = brdf_path[tile_id]  
    # Extract pixels
    brick = pd.DataFrame()
    with rasterio.open("//orange/"+brdf_path.iloc[0]) as src:
        for ii in range(itc.shape[0]):
            # Extract feature shapes and values from the array.
            ith_geoms = [mapping(geoms[ii])]
            clip, affine = mask(src, ith_geoms, crop=True)
            cl,rw = clip.shape[1:3]
            clip = clip.transpose(1,2,0).reshape(cl*rw,369)
            clip = pd.DataFrame(clip)
            clip["individualID"] = ii
            brick = brick.append(pd.DataFrame(data = clip), ignore_index=True)
    brick.to_csv(out_pt+"/"+tile+"_brdf_itc.csv")   