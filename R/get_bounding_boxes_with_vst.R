vst = readr::read_csv("/Users/sergiomarconi/Documents/Data/Surveys/VST/vst_field_data.csv")
outdir = "~/Documents/Data/vst_df_top/"

vst_ids = vst %>% select(siteID, plotID,individualID, itcEasting, itcNorthing, utmZone, taxonID, stemDiameter, elevation, plantStatus, canopyPosition) %>% 
  group_by(individualID) %>%  slice(1)

plots = list.files("~/Documents/Data/deepForest_plot_crops/", full.names = T, pattern = ".shp") %>% data.frame
colnames(plots) = "pt"
results = list()
for(ii in unique(vst_ids$plotID)){
  tryCatch({
  f= plots %>%dplyr::filter(str_detect(pt, ii))
  f = tail(f, n=1)
  deep_boxes = sf::read_sf(f$pt)
  vst_ = vst_ids %>% filter(plotID == ii) 
  vst_ = sf::st_as_sf(vst_, coords = c("itcEasting", "itcNorthing"), crs = sf::st_crs(deep_boxes))
  results[[ii]] = sf::st_join(deep_boxes, vst_)
  sf::write_sf(results[[ii]], paste(outdir, "/deep_vst_", ii,".shp", sep=""))
  },error = function(e) {paste(f, "error")})
}
final2 = do.call(rbind.data.frame, results)
write_csv(final2, "~/Documents/Data/Data_products/deepForest_full_vst.csv")
#!test %in% test[duplicated(test)]
unique_id = final2 %>% filter(!individualID %in% final2$individualID[duplicated(final2$individualID)])
shared_id = final2 %>% filter(individualID %in% final2$individualID[duplicated(final2$individualID)])%>%
  filter(!is.na(individualID))

good_ones = shared_id %>% data.frame %>% select(individualID, taxonID) %>% unique
good_ones = good_ones$individualID
aa = shared_id %>% filter(individualID %in% good_ones) %>% group_by(individualID) %>% slice(1)
final_uniques = rbind.data.frame(unique_id, aa)
write_csv(final_uniques, "~/Documents/Data/Data_products/deepForest_full_uniques.csv")
write_csv(shared_id, "~/Documents/Data/Data_products/deepForest_full_need_check.csv")
