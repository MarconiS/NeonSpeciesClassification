#plot stack per site
library(raster)
library(tidyvrse)

get_epsg_from_utm <- function(utm){
  utm <-  substr(utm,1,nchar(utm)-1)
  if(as.numeric(utm)<10)utm <- paste('0', utm, sep="")
  epsg <- paste("326", utm, sep="")
  return(epsg)
}

list_plots = list.files("~/Documents/Data/RS/plots/bdrf/", full.names = T) 
good_trees = readr::read_csv("~/Documents/Data/NEON/VST/vst_explicit_canopy_position.csv")
good_trees = sf::st_as_sf(good_trees, coords = c("itcLongitude", "itcLatitude"), crs = 4326)

full_dataset = list()
tkn = 0
for (plt in list_plots){
  brk = brick(plt)
  pltid = stringr::str_sub(plt, start=-12, end =-5)
  trees = good_trees %>% filter(plotID %in% pltid)
  if(nrow(trees) > 0){
    trees = sf::st_transform(trees, crs = as.integer(get_epsg_from_utm(unique(trees$utmZone))))
    trees = sf::st_crop(trees, extent(brk))
  }
  if(nrow(trees) > 0)
    for(itc in 1:nrow(trees)){
      tkn = tkn + 1
      foo = extract(brk, trees[itc,], buffer = 1, df = T) %>% data.frame
      colnames(foo) = c("individualID", paste("band", 1:369, sep="_"))
      foo$individualID = trees[["individualID"]][itc]
      full_dataset[[tkn]] = foo
    }
}


final_dataset = do.call(rbind.data.frame, full_dataset)
write_csv(final_dataset, "~/Documents/Data/NEON/VST/vst_top_bf1_reflectance.csv")

