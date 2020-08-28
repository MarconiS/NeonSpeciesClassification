#plot stack per site
library(raster)
library(tidyvrse)

get_epsg_from_utm <- function(utm){
  utm <-  substr(utm,1,nchar(utm)-1)
  if(as.numeric(utm)<10)utm <- paste('0', utm, sep="")
  epsg <- paste("326", utm, sep="")
  return(epsg)
}

list_plots = list.files("/Volumes/Stele/brdf_plots/", full.names = T, pattern = "East")
good_trees = readr::read_csv("~/Documents/Data/Surveys/VST/vst_field_data.csv")
good_trees = sf::st_as_sf(good_trees, coords = c("itcLongitude", "itcLatitude"), crs = 4326)
good_trees = good_trees %>% filter(individualID %in% missing_) %>% group_by(individualID) %>% slice(1)
good_trees$plotID %>% unique
dplyr::filter(list_plots, grepl(good_trees$plotID %>% unique, pt))


full_dataset = list()
tkn = 0
for (plt in ls_check){
  brk = brick(plt)
  pltid = stringr::str_sub(plt, start=28, end =35)
  trees = good_trees %>% filter(plotID %in% pltid)
  if(nrow(trees) > 0){
    trees = sf::st_transform(trees, crs = as.integer(get_epsg_from_utm(unique(trees$utmZone))))
    trees = sf::st_crop(trees, extent(brk))
    if(nrow(trees) > 0){
      
      for(itc in 1:nrow(trees)){
        tkn = tkn + 1
        foo = raster::extract(brk, trees[itc,], buffer = 2, df = T) %>% data.frame
        colnames(foo) = c("individualID", paste("band", 1:367, sep="_"))
        foo$individualID = trees[["individualID"]][itc]
        full_dataset[[tkn]] = foo
      }
    }else{
      print(pltid)
    }
  }
}


final2_dataset = do.call(rbind.data.frame, full_dataset)
write_csv(final2_dataset, "~/Documents/Data/Surveys//VST/vst_Aug_bf2_reflectance.csv")

