library(sf)
library(tidyverse)
clean_spectra <- function(brick){
  # filter for no data 
  mask = brick > 10000
  brick[mask] <-  NA
  mask = brick == -9999
  brick[mask] <-  NA
  
  #filter for greennes and shadows
  ndvi <- (brick[,"band_90"]- brick[,"band_58"])/(brick[,"band_58"] + brick[,"band_90"]) <0.3
  nir860 <- (brick[,"band_96"] + brick[,"band_97"])/2 < 0.01
  mask = as.logical(ndvi | nir860)
  mask[is.na(mask)] = T
  brick[mask,] = NA
  rm(mask, ndvi, nir860)
  brick = brick[,15:365]
  normMat <- sqrt(apply(brick^2,FUN=sum,MAR=1, na.rm=TRUE))
  normMat <- matrix(data=rep(normMat,ncol(brick)),ncol=ncol(brick))
  brick=brick/normMat
  rm(normMat)
  
  #filter for known artifacts
  cnd = (brick[,"band_312"] > 0.03)
  cnd[is.na(cnd)] = T
  #idx <- which(apply(cnd, 1, any))
  brick[cnd,] = NA
  
  cnd = (brick[,24:45] > 0.03)
  idx <- (apply(cnd, 1, any))
  if(length(idx) !=0){
    idx[is.na(idx)] = T
    brick[idx,] = NA
  }
  cnd = (brick[,195:200] > 0.043)
  idx <- (apply(cnd, 1, any))
  if(length(idx) !=0){
    idx[is.na(idx)] = T
    brick[idx,] = NA
  }
  rm(cnd,idx)
  
  # # save pixel positions
  good_pix = !is.na(brick)
  good_pix = (apply(good_pix, 1, all))
  # 
  brick = brick[complete.cases(brick),]
  return(list(refl = brick, good_pix = good_pix))
}


#dataset = readr::read_csv("../NeonSpeciesClassification/dimensionality_reduction/hsi_appended.csv")
hpca_predict <- function(dataset, kld_obj, method = "hist"){
  hsi = dataset %>% dplyr::select(matches("band"))
  kld_array <- kld_obj$bands_grouping
  #perform PCA for each group and select first component
  grp = cbind.data.frame(kld_array, t(hsi))
  if( method=='hist'){
    #get min, max and auc
    kld_refl = list()
    for(gp in unique(kld_array)){
      pcx = grp %>% dplyr::filter(kld_array == gp) 
      pcx = pcx[-1]%>% t 
      min_sband = apply(pcx, 1, min)
      max_sband = apply(pcx, 1, max)
      # scale and sum (value-min)/(max-min)
      pcxgrp =  kld_obj$compression[,gp] #c(min(pcx), max(pcx))
      auc_sband = apply(pcx, 1, function(x)(x-pcxgrp[1])/(pcxgrp[2]-pcxgrp[1]))
      auc_sband =  apply(pcx, 1, sum)
      auc_sband = cbind(min_sband, max_sband, auc_sband)
      colnames(auc_sband) = paste(c("min_kl_", "max_kl_", "auc_kl_"), gp, sep="")
      kld_refl[[gp]] = auc_sband
    }
    kld_refl = do.call(cbind.data.frame, kld_refl)
  }
  return(kld_refl)
}

#extract pixels around centroid
pt = "////orange/idtrees-collab/hsi_brdf_corrected/brdf_to_csv/"
out_dir= "/ufrc/ewhite/s.marconi/weak_classifiers/pred_indir/"
tl = list.files(pt, pattern = ".csv")
for(tile in tl){
  #tile = "GUAN_724000_1987000_brdf_itc.csv"
  itcc = readr::read_csv(paste(pt, tile, sep=""))
  #get_domain = readr::read_csv("/ufrc/ewhite/s.marconi/weak_classifiers/get_domains.csv")
  colnames(itcc) = c("dn", paste("band", 1:369, sep="_"), "individualID")
  itcc = data.frame(itcc)
  data = itcc %>% select(-one_of("dn", "individualID"))
  data =clean_spectra(itcc)
  #itcs_pixels = itcs_pixels[data$good_pix,]
  data = cbind.data.frame(itcc$individualID[data$good_pix], data$refl)
  #load data reduction 
  cfc_reduced_dims = readRDS("../traitsMaps/indir/kld_hist_june_30.rds")   #"/ufrc/ewhite/s.marconi/weak_classifiers/kld_hist_all_30.rds")
  foo = hpca_predict(data, cfc_reduced_dims)
  individualID = data[1]
  colnames(individualID)="individualID"
  siteID = substr(tile, 1, 4)
  domainID = as.character(unlist(get_domain[get_domain["siteID"] == siteID, "domainID"]))
  
  write_csv(cbind.data.frame(individualID, siteID, domainID, foo), paste(out_dir, "kl30", tile, sep="/"))
}
#write_csv(data.frame(itcs_pixels), "~/Documents/Data/HS/HARV_725000_4696000.csv")
#itcs_pixels = read_csv("//Volumes/Stele/example_itcextract.csv")
#run prediction
# 
# 
# 
# 
# #append reult
# preds = read_csv("./weak_label/pred_out/HARV_725000_4696000.csv")
# traits = read_csv("~/Documents/Data/HS/HARV_725000_4696000_traits.csv")
# colnames(preds) = c("id", "individualID", "taxonID")
# tile_itc$individualID = 1:nrow(tile_itc)
# species_predictions = inner_join(tile_itc, preds)
# traits_predictions = inner_join(tile_itc, traits)
# write_sf(species_predictions, "./weak_label/pred_out/HARV_725000_4696000.shp")
# write_sf(traits_predictions, "~/Documents/Data/HS/traits_HARV_725000_4696000.shp")
