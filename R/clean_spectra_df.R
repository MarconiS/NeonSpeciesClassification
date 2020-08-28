clean_spectra <- function(brick, ndvi = 0.5, nir = 0.2){
  # filter for no data
  brick = brick %>% ungroup %>% dplyr::select(contains("band"))
  mask1 = apply(brick, 1, function(x)all(x>-1))
  mask2 = apply(brick, 1, function(x)all(x<10000))
  brick[!as.logical(mask2), ] = NA
  brick[!as.logical(mask1), ] = NA
  brick[brick<0]=0
  brick[,30:40][brick[,30:40]==0] = NA
  
  #filter for greennes and shadows
  ndvi <- (brick[,"band_90"]- brick[,"band_58"])/(brick[,"band_58"] + brick[,"band_90"]) <ndvi
  nir860 <- (brick[,"band_96"] + brick[,"band_97"])/20000 < nir
  mask = as.logical(ndvi * nir860)
  mask[is.na(mask)] = T
  brick[mask,] = NA
  rm(mask, ndvi, nir860)
  brick = brick %>% dplyr::select(one_of(paste("band", 15:360, sep="_")))
  normMat <- sqrt(apply(brick^2,FUN=sum,MAR=1, na.rm=TRUE))
  normMat <- matrix(data=rep(normMat,ncol(brick)),ncol=ncol(brick))
  brick=brick/normMat
  rm(normMat)
  
  # #check range of spectra
  # max_ideal =  apply(brick[complete.cases(brick),],
  #                    MARGIN = 2, function(x)quantile(x, 0.996))
  # min_ideal =  apply(brick[complete.cases(brick),],
  #                    MARGIN = 2, function(x)quantile(x, 0.004))
  # 
  # plot(min_ideal)
  # plot(max_ideal)
  # #filter for outliers: too bright
  # cnd = apply(brick, 1,function(x)(x > max_ideal))
  # idx <- (apply(data.frame(cnd), 2, any))
  # if(length(idx) !=0){
  #   idx[is.na(idx)] = T
  #   brick[idx,] = NA
  # }
  # #filter for outliers: too dark
  # cnd = apply(brick, 1,function(x)(x < min_ideal))
  # idx <- (apply(data.frame(cnd), 2, any))
  # if(length(idx) !=0){
  #   idx[is.na(idx)] = T
  #   brick[idx,] = NA
  # }
  # rm(cnd,idx)
  
  # # save pixel positions
  good_pix = !is.na(brick)
  good_pix = (apply(good_pix, 1, all))
  #
  brick = brick[complete.cases(brick),]
  return(list(refl = brick, good_pix = good_pix))
}

hsi = clean_spectra(final_final)
brick = final_final[hsi$good_pix,]
brick = unique(brick)
min_ideal = apply(brick[,2:200], 2, min)
max_ideal = apply(brick[,2:200], 2, max)

ids = brick$band_34 == 11
par(mar=c(1,1,1,1))
plot(unlist(brick[ids,2:350]))
vst = readr::read_csv("./weak_label/indir/csv/vst_field_data.csv") %>%
  select(individualID, taxonID, siteID, domainID, elevation, stemDiameter, height, growthForm, plantStatus, canopyPosition, nlcdClass, Easting, Northing, itcLongitude, itcLatitude) %>%
  group_by(individualID) %>% top_n(1, wt = "height") %>% slice(1)
colnames(brick)[1] = "individualID"
brick = cbind.data.frame(brick[1],hsi$refl)
brick = left_join(brick, vst) %>% unique

#ids_train = read_csv("~/Documents/Data/forBen/testing_data.csv") %>% select(individualID, taxonID) %>% unique

brick$taxonID[brick$taxonID=="ABIES"] = NA #"ABBA"
brick$taxonID[brick$taxonID=="ACRUR"] = "ACRU"
brick$taxonID[brick$taxonID=="ACSAS"] = "ACSA3"
brick$taxonID[brick$taxonID=="BEPAP"] = "BEPA"
brick$taxonID[brick$taxonID=="AQUIFOSPP"] = "AQUIFO"
brick$taxonID[brick$taxonID=="AQUIFO"] = NA
brick$taxonID[brick$taxonID=="BETUL"] = NA#"BELE"
brick$taxonID[brick$taxonID=="FRAXI"] = NA#"FRAM"
brick$taxonID[brick$taxonID=="JUNIP"] = "JUVI"
brick$taxonID[brick$taxonID=="MAGNO"] = NA #"MAFR"
brick$taxonID[brick$taxonID=="PSMEM"] = "PSME"
brick$taxonID[brick$taxonID=="QUHE"] = "QUHE2"
brick$taxonID[brick$taxonID=="PRSES"] = "PRSE2"
brick$taxonID[brick$taxonID=="SALIX"] = NA #"SASC"
brick$taxonID[brick$taxonID=="2PLANT"] = NA
brick$taxonID[brick$taxonID=="OXYDE"] = NA
brick$taxonID[brick$taxonID=="HALES"] = NA
brick$taxonID[brick$taxonID=="PINUS"] = NA
brick$taxonID[brick$taxonID=="QUERC"] = NA
brick$taxonID[brick$taxonID=="PICEA"] = NA
brick$taxonID[brick$taxonID=="PICOL"] = "PICO"

if(onlyTrees){
  list_trees = read_csv("/Users/sergiomarconi/Documents/Data/Surveys/VST/list_of_species_lifeform.csv") %>%
    filter(lifeForm %in% c("T", "TS"))
  fair_far = read_csv("~/Documents/Data/brdf_partial_dataset.csv")
  brick = brick %>% filter(taxonID %in% list_trees$taxonID)
  brick = brick %>% filter(taxonID %in% fair_far$taxonID)
}

#remove shaded and keep only dominant NA
taxa_missing = c("SWMA2","PIRU", "QUCH", "PIRU", "PSMEM","PIJE","BEGL/BENA")
dominant_na = brick %>% filter(is.na(canopyPosition)) %>%
  filter(taxonID %in% taxa_missing)
brick =  brick %>% filter(!canopyPosition %in% c("Full shade", "Mostly shaded"), !is.na(canopyPosition))
brick = rbind.data.frame(dominant_na, brick)


brick = brick[complete.cases(brick[,2:350]),]
check_species = brick %>% select(individualID, taxonID) %>% unique
check_species = check_species$taxonID %>% table %>% data.frame
colnames(check_species)[1] <- "taxonID"
rare_species = check_species %>% filter(Freq < 6 ) %>% select(taxonID)
brick = brick %>% filter(!taxonID %in% rare_species$taxonID)

#clean and scale dataset
#apply dimensionality reduction
#cfc_reduced_dims = hpca(brick2, npcs = nComp, method = 'hist')
foo = brick[20:358]
foo[foo==0]=NA
foo=foo[complete.cases(foo),]
foo = scale(foo)
write_csv(data.frame(foo), "./foo_kld.csv")



write_csv(brick, "./weak_label/indir/csv/dataset_marconi2020_brdf_centers.csv")
allometry_baseline = brick[c(1,368:382)] %>% unique
write_csv(allometry_baseline, "./weak_label/indir/csv/dataset_marconi2020_baseline.csv")
