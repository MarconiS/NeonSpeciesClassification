ch = vst %>% select(individualID, taxonID, Easting, Northing) %>% 
  group_by(individualID) %>%slice(1)

ch = ch %>% filter(individualID %in% unique(brick2$individualID))
dist_e = dist(as.integer(ch$Easting)) %>% as.matrix
dist_n = dist(ch$Northing) %>% as.matrix
very_close = (dist_e <2) & (dist_n < 2)
pairs_mat = matrix(T, nrow = nrow(dist_e), ncol = ncol(dist_e))
rownames(pairs_mat) = colnames(pairs_mat) =  ch$taxonID
pairs_mat[!very_close] = F

paired = t(t(apply(pairs_mat, 1, function(u) paste( names(which(u)), collapse=", " ))))
paired 

get_confused = list()
which_confused = rep(NA, nrow(paired))
for(ii in 1:nrow(paired)){
  foo = strsplit(paired[ii], split = ", ") %>% unlist %>%unique
  get_confused[[ii]] = foo
  if(length(foo) >1) {which_confused[ii]=1}
  else {which_confused[ii]=0}
}
foo = get_confused[!!which_confused]
ids_too_mixed = ch$individualID[!!which_confused] %>% unique
fbf = brick2 %>% filter(!individualID %in% ids_too_mixed)
taxa_missing = c("SWMA2","PIRU", "QUCH", "PIRU", "PSMEM","PIJE","BEGL/BENA")
fbrdf = fbf %>% filter(canopyPosition %in% unique(fbf$canopyPosition)[c(1,2,5)])
fbrdf2 = fbf %>% filter(is.na(canopyPosition)) %>% filter(taxonID %in% taxa_missing)
fbrdf = rbind.data.frame(fbrdf, fbrdf2)

write_csv(fbrdf, "~/Documents/Data/brdf_partial_dataset.csv")
