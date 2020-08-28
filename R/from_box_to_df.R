vst2 = readr::read_csv("/Users/sergiomarconi/Documents/Data/Surveys/VST/vst_field_data.csv")
indir = "~/Documents/Data/vst_df_top/"
tiff_pt = "/Volumes/Stele/brdf_plots"
outdir =  "/Volumes/Stele/brdf_itc"
library(tidyverse)
plots = list.files(indir, full.names = T, pattern = ".shp") %>% data.frame
spectral_dataset = list()
token = 0
for(ii in unlist(plots)){
  tryCatch({
    deep_boxes = sf::read_sf(ii)
    deep_boxes = deep_boxes %>% filter(!is.na(indvdID))
    #get tile info
    tile_info = str_sub(ii,-12,-5)
    plot_tif = list.files(tiff_pt, full.names = T, pattern = tile_info) %>% data.frame
    brk = raster::brick(unlist(plot_tif))
    for(jj in 1:nrow(deep_boxes)){
      token = token+1
      itc = deep_boxes[jj,]
      itc = sf::st_buffer(itc, 0)
      crop = raster::crop(brk, itc)
      tmp_df = raster::as.data.frame(crop)
      colnames(tmp_df) = paste("band", 1:ncol(tmp_df), sep="_")
      itc = itc %>% data.frame %>% select(siteID, plotID, indvdID, taxonID, height, area, stmDmtr, elevatn, plntStt, cnpyPst)
      spectral_dataset[[token]] = cbind(tmp_df, itc)
    }
  },error = function(e) {paste(ii, "error")})
}
final_dataset = do.call(rbind.data.frame, spectral_dataset)
