# get reduction from all training images
#append all images, clear them from bad pixels, define (1) the endmembers, (2) the klds
list_plots = list.files("~/Documents/Data/RS/hsi/", pattern = ".tif", full.names = T)
hsi_append = list()
for(ii in 1:length(list_plots)){
  pt = list_plots[ii]
  tmp = raster::brick(pt)
  tmp= raster::as.data.frame(tmp)
  colnames(tmp) = paste("band", 1:369, sep="_")
  hsi_append[[ii]] = tmp
}
data_dimred = do.call(rbind.data.frame, hsi_append)
readr::write_csv(data_dimred, "./dimensionality_reduction/hsi_appended.csv")
