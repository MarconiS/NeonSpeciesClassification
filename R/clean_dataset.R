#get only renown top of the crown data
library(tidyverse)
vst <- readr::read_csv("./weak_label/indir/csv/vst_field_data.csv") 
known_top = vst %>%  filter(canopyPosition %in% c("Full sun", "Partially shaded", "Open grown")) %>%
  select(individualID, eventID, itcLongitude, itcLatitude, siteID, plotID, plantStatus,canopyPosition,
         taxonID, maxCrownDiameter, ninetyCrownDiameter, height, stemDiameter) %>%
  group_by(individualID) %>%
  sample_n(1)

write_csv(known_top, "./weak_label/indir/csv/top_of_the_canopy.csv")

#only for those sites where I have traits for
sites_traits = read_csv('./weak_label/indir/csv/cfc_field_data.csv') %>% select(siteID) %>% unique
treated_sites = known_top %>% filter(siteID %in% unlist(sites_traits))
treated_sites$individualID = substring(treated_sites$individualID, 14)
treated_sites$treeID = substring(treated_sites$individualID, 6)
write_csv(treated_sites, "./weak_label/indir/csv/top_of_the_canopy.csv")
foo = sf::read_sf("./weak_label/indir/shp/HARV.shp")
foo = foo %>% filter(treeID %in% treated_sites$treeID)
sf::write_sf(foo, "./weak_label/indir/shp/vst_polygons.shp")
#meuse_sf = st_as_sf(treated_sites, coords = c("x", "y"), crs = 28992, agr = "constant")
treated_sites = sf::st_as_sf(treated_sites, coords = c("itcLongitude", "itcLatitude"), crs = 4326)
sf::write_sf(treated_sites, "./weak_label/indir/shp/vst_ground.shp")
