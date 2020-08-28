#load vegetation structure
vst = readr::read_csv("////blue/ewhite/s.marconi/NeonSpeciesClassification/weak_label/indir/csv/vst_field_data.csv")
bboxes = "////orange/idtrees-collab/draped/"
bboxes = "////orange/idtrees-collab/hsi_brdf_corrected/corrHSI/"
outdir = "/orange/ewhite/s.marconi/brdf_plots/"


vst$taxonID %>% unique
target_plots = readr::read_csv("//blue/ewhite/s.marconi/NeonSpeciesClassification/weak_label/plots_to_increase_x.csv")
target_plots =  substr(target_plots$ls_check, 28, 35)

vst_ids = vst %>% select(siteID, plotID,easting, northing) %>% group_by(siteID, plotID) %>%
  summarize_all(mean) %>% unique
vst_ids = vst_ids %>% filter(plotID %in% target_plots)
vst_ids$tileE = as.integer(vst_ids$easting/1000)*1000
vst_ids$tileN = as.integer(vst_ids$northing/1000)*1000

#vst_ids = vst_ids %>% filter(siteID == "HARV")
for(ii in 1:nrow(vst_ids)){
  tryCatch({
    #_reflectance.tif
    pattern = paste(vst_ids$tileE[ii],"_", vst_ids$tileN[ii], "_reflectance.tif", sep="") #"_image.shp", sep="")
    ls_tile = list.files(bboxes, pattern = pattern, full.names = T) %>% data.frame
    colnames(ls_tile) = "pt"
    ls_tile = ls_tile %>%dplyr::filter(str_detect(pt, vst_ids$siteID[ii]))
    for(get_tile in ls_tile$pt){
      #tile = sf::read_sf(as.character(get_tile))
      tile = raster::brick(as.character(get_tile))
      file_id = strsplit(as.character(get_tile), split = "/")
      file_id = tail(file_id[[1]], n=1)
      if(!file.exists(paste(outdir, "/",vst_ids$plotID[ii],"_drAped_", file_id, sep=""))){
        plot_crop = vst_ids[ii,]
        plot_crop = sf::st_as_sf(plot_crop, coords = c("easting", "northing"), crs = sf::st_crs(tile))
        plot_crop = sf::st_buffer(plot_crop, 20)
        #crop = sf::st_crop(tile, plot_crop)
        crop = raster::crop(tile, plot_crop)
        #sf::write_sf(crop, paste(outdir, "/",vst_ids$plotID[ii],"_draped_", file_id, sep=""))
        raster::writeRaster(crop, paste(outdir, "/East_",vst_ids$plotID[ii],"_draped_", file_id, sep=""))
      }
    }
  },error = function(e) {paste(pattern, "error")})
}