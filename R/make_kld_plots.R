list_plots = list.files("~/Documents/GitHub/neonVegWrangleR/outdir/plots/hsi/", pattern = ".tif", full.names = T)
hsi_append = list()
kld_obj = readRDS("../traitsMaps/indir/kld_bands18.rds")
for(ii in 1:length(list_plots)){
  pt = list_plots[ii]
  brick = raster::brick(pt) 
  tile_dim = dim(brick)
  brick = raster::as.matrix(brick)
  colnames(brick) = paste("band", 1:369, sep="_")
  
  #clean tile  
  mask = brick > 10000
  brick[mask] <-  NA
  mask = brick == -9999
  brick[mask] <-  NA
  
  ndvi <- (brick[,"band_90"]- brick[,"band_58"])/(brick[,"band_58"] + brick[,"band_90"]) <0.3
  nir860 <- (brick[,"band_96"] + brick[,"band_97"])/20000 < 0.1
  mask = as.logical(ndvi | nir860)
  brick[mask,] = NA
  rm(mask, ndvi, nir860)
  brick = brick[,15:365]
  normMat <- sqrt(apply(brick^2,FUN=sum,MAR=1, na.rm=TRUE))
  normMat <- matrix(data=rep(normMat,ncol(brick)),ncol=ncol(brick))
  brick=brick/normMat
  rm(normMat)
  
  cnd = (brick[,"band_312"] > 0.025)  
  #idx <- which(apply(cnd, 1, any))
  brick[cnd,] = NA
  
  cnd = (brick[,24:45] > 0.04)
  idx <- (apply(cnd, 1, any))
  if(length(idx) !=0){
    brick[idx,] = NA
  }
  cnd = (brick[,100:200] > 0.15)
  idx <- (apply(cnd, 1, any))
  if(length(idx) !=0){
    brick[idx,] = NA
  }
  rm(cnd,idx)
  
  # save pixel positions
  good_pix = !is.na(brick)
  good_pix = (apply(good_pix, 1, all))
  if(sum(good_pix) > 4){
    brick = brick[complete.cases(brick),]
    
    # reduce into 1st PC of the KLD 
    brick = cbind.data.frame(kld_obj$bands_grouping, t(brick))
    colnames(brick)[1] = "kld_array"
    kld_refl = list()
    #loop through groups and create a PCA for each
    for(gp in unique(kld_obj$bands_grouping)){
      pcx = brick %>% dplyr::filter(kld_array == gp) #%>% t %>% prcomp()
      pcxgrp = predict(kld_obj$pcas[[gp]], newdata = t(pcx))
      kld_refl[[gp]] = pcxgrp[,1]
    }
    brick = do.call(cbind.data.frame, kld_refl)
    rm(kld_refl, pcx, pcxgrp)
    colnames(brick) = paste("kd", 1:ncol(brick), sep="_")
    
    # add accessory features
    brick = brick[-1,]
    
    lyr = (matrix(NA, tile_dim[1] *tile_dim[2],ncol(brick)))
    lyr[good_pix,] = as.matrix(brick)
    dim(lyr) = c(tile_dim[1], tile_dim[2],ncol(brick))
    
    # 
    dat = raster::brick(pt) 
    lyr = raster::brick(lyr, xmn=dat@extent[1], xmx=dat@extent[2], #nl = 9,
                        ymn=dat@extent[3], ymx=dat@extent[4], crs=dat@crs, transpose=FALSE)
    
    raster::writeRaster(lyr, 
                        paste("./dimensionality_reduction/kld/",
                              substr(pt, nchar(pt)-12+1, nchar(pt)-4), "_kld.tiff", sep =""),  overwrite=TRUE)
  }
}
