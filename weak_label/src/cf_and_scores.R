#get scores per domain
library(tidyverse)
library(caret)
get.conf.stats <- function(cm) {
  out <- vector("list", length(cm))
  for (i in seq_along(cm)) {
    x <- cm[[i]]
    tp <- x$table[x$positive, x$positive] 
    fp <- sum(x$table[x$positive, colnames(x$table) != x$positive])
    fn <- sum(x$table[colnames(x$table) != x$positie, x$positive])
    # TNs are not well-defined for one-vs-all approach
    elem <- c(tp = tp, fp = fp, fn = fn)
    out[[i]] <- elem
  }
  df <- do.call(rbind, out)
  rownames(df) <- unlist(lapply(cm, function(x) x$positive))
  return(as.data.frame(df))
}
get.micro.f1 <- function(cm) {
  cm.summary <- get.conf.stats(cm)
  tp <- sum(cm.summary$tp)
  fn <- sum(cm.summary$fn)
  fp <- sum(cm.summary$fp)
  pr <- tp / (tp + fp)
  re <- tp / (tp + fn)
  f1 <- 2 * ((pr * re) / (pr + re))
  return(f1)
}
list_of_species = read_csv("~/Documents/Data/NEON/VST/list_of_species_lifeform.csv")
pairs = readr::read_csv("./weak_an__upperTrees_final_kld_pairs.csv")
probabilities = readr::read_csv("./weak_an__upperTrees_final_kld_probabilities.csv")
tree_species = list_of_species %>% filter(lifeForm %in% c("T"))
# 
# colnames(probabilities)[which(probabilities[1,-(1:3)] == max(probabilities[1,-(1:3)]))]
colnames(pairs)=c("id", "individualID", "obs", "pred")
pairs$domainID = substr(pairs$individualID, 10,12)
pairs$siteID = substr(pairs$individualID, 14,17)

sp_in_dat = pairs$obs %>% data.frame
colnames(sp_in_dat) = "taxonID"
life_forms = inner_join(sp_in_dat, list_of_species)

vst = readr::read_csv("~/Documents/Data/NEON/VST/vst_field_data.csv")
vst = vst %>% group_by(individualID) %>% slice(1) %>% ungroup
vst = vst %>% filter(siteID %in% unique(pairs$siteID))
vst[vst$taxonID=="PSMEM","taxonID"] = "PSME"

species_per_site = vst %>% filter(taxonID %in% tree_species$taxonID)  
tot_species = species_per_site %>% select(taxonID, siteID) %>% group_by(siteID) %>% table
data = read_csv("./weak_label/indir/csv/brdf_T_hist.csv")

#want to check how many species out of the total we have in the dataset for each site
spst = spdt =summary_freqs =  needed_missing = list()
for(ii in 1:ncol(tot_species)){
  sp_in_site =  names(which(tot_species[,ii] >0))
  sort(unique(data$taxonID))
  sp_in_dat = (sp_in_site) %in% unique(data$taxonID)
  needed_missing[[ii]] = sp_in_site[!sp_in_dat]
  sp_frac = sum(sp_in_dat)/length(sp_in_site)
  sp_abundance = tot_species[sp_in_site,ii] 
  itc_frac= sum(sp_abundance[sp_in_dat])/sum(sp_abundance)
  summary_freqs[[ii]] = c(colnames(tot_species)[ii], sp_frac, itc_frac)
  spst[[ii]] = sp_in_site
  spdt[[ii]] = sp_in_dat
}
summary_freqs = do.call(rbind.data.frame, summary_freqs) %>% data.frame
colnames(summary_freqs) = c("siteID", "species_fraction", "abundance_fraction")
summary_freqs$abundance_fraction = as.numeric(as.character(summary_freqs$abundance_fraction))
list_missing = do.call(c, needed_missing)
cm = list()
microF1 = list()
for(dm in unique(pairs$domainID)){
  dm_dt = pairs %>% filter(domainID == dm)
  
  dm_dt$obs = factor(dm_dt$obs, levels = unique(data$taxonID))
  dm_dt$pred = factor(dm_dt$pred, levels = unique(data$taxonID))
  cmdm = confusionMatrix(dm_dt$obs, dm_dt$pred)
  print(cmdm$overall)
  #mcm = as.matrix.data.frame(cmdm$table)
  #rownames(mcm) = colnames(mcm) = colnames(cmdm$table)
  microF1[[dm]] <- cmdm$overall[1]
  cm[[dm]] = cmdm
}
microF1_dom = unlist(microF1)
cm_dom = cm

cm = list()
microF1 = list()
for(dm in unique(pairs$siteID)){
  dm_dt = pairs %>% filter(siteID == dm)
  
  dm_dt$obs = factor(dm_dt$obs, levels = unique(data$taxonID))
  dm_dt$pred = factor(dm_dt$pred, levels = unique(data$taxonID))
  cmdm = confusionMatrix(dm_dt$obs, dm_dt$pred)
  print(cmdm$overall)
  #mcm = as.matrix.data.frame(cmdm$table)
  #rownames(mcm) = colnames(mcm) = colnames(cmdm$table)
  microF1[[dm]] <- cmdm$overall[1]
  cm[[dm]] = cmdm
}
microF1_site= unlist(microF1)

site_pairs = pairs %>% group_by(siteID) %>% select(obs) %>% table 
sp_per_site = apply(site_pairs, 1, function(x)(sum(x>0)))
entries_per_site = apply(site_pairs, 1, function(x)(sum(x)))

dom_pairs = pairs %>% group_by(domainID) %>% select(obs) %>% table 
sp_per_domain = apply(dom_pairs, 1, function(x)(sum(x>0)))
entries_per_domain = apply(dom_pairs, 1, function(x)(sum(x)))

#domain analysis
dm = cbind.data.frame(sp_per_domain, entries_per_domain, microF1_dom)
dm$Domain = rownames(dm)
ggplot(dm, aes(x = Domain, y = entries_per_domain)) + geom_bar(stat="identity") + theme_bw()

cor(dm$sp_per_domain, dm$microF1_dom)
