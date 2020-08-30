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
list_of_species = read_csv("~/Documents/Data/Surveys/VST/list_of_species_lifeform.csv")
pairs = readr::read_csv("./weak_label/mods/weak_an__final80_kld_pairs.csv")
pairs = readr::read_csv("./weak_an__final80_family_predictions.csv")
probabilities = readr::read_csv("./weak_label/mods/weak_an__final80_kld_probabilities.csv")
tree_species = list_of_species %>% filter(lifeForm %in% c("T", "TS"))
# 
# colnames(probabilities)[which(probabilities[1,-(1:3)] == max(probabilities[1,-(1:3)]))]
colnames(pairs)=c("id", "individualID", "obs", "pred")
pairs$domainID = substr(pairs$individualID, 10,12)
pairs$siteID = substr(pairs$individualID, 14,17)

sp_in_dat = pairs$obs %>% data.frame
colnames(sp_in_dat) = "taxonID"
life_forms = inner_join(sp_in_dat, list_of_species)

vst = readr::read_csv("~/Documents/Data/Surveys/VST/vst_field_data.csv")

vst = vst %>% group_by(individualID) %>% slice(1) %>% ungroup
vst = vst %>% filter(siteID %in% unique(pairs$siteID))
vst[vst$taxonID=="PSMEM","taxonID"] = "PSME"
vst[vst$taxonID=="ACRUR","taxonID"] = "ACRU"
vst[vst$taxonID=="ACSAS","taxonID"] = "ACSA3"
vst[vst$taxonID=="BEPAP","taxonID"] = "BEPA"
vst[vst$taxonID=="AQUIFOSPP","taxonID"] = "AQUIFO"
vst[vst$taxonID=="JUNIP","taxonID"] = "JUVI"
vst[vst$taxonID=="QUHE","taxonID"] = "QUHE2"
vst[vst$taxonID=="PRSES","taxonID"] = "PRSE2"
vst[vst$taxonID=="PICOL","taxonID"] = "PICO" 
species_per_site = vst %>% filter(taxonID %in% tree_species$taxonID) 
tot_species = species_per_site %>% select(taxonID, siteID)  %>% group_by(siteID) %>% table
data = read_csv("./weak_label/indir/csv/test_plus_2020_brdf_centers.csv")

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
sp_st = lapply(1:23, function(x) length(spst[[x]]))
sp_fr = lapply(1:23, function(x) sum(spdt[[x]]))
sp_st = unlist(sp_st) %>% data.frame
colnames(sp_st) = "alpha"
sp_st["siteID"] = colnames(tot_species)
sp_st["indata"] = unlist(sp_fr)
sp_st = reshape2::melt(sp_st, "siteID")
ggplot(sp_st, aes(fill=variable, y=value, x=siteID)) + 
  geom_bar(position="dodge", stat="identity") + 
  theme_bw() + theme(axis.text.x = element_text(size=14, angle=45, hjust=1, vjust=1)) 
  
summary_freqs = do.call(rbind.data.frame, summary_freqs) %>% data.frame
colnames(summary_freqs) = c("siteID", "species_fraction", "abundance_fraction")
summary_freqs$abundance_fraction = as.numeric(as.character(summary_freqs$abundance_fraction))
summary_freqs$species_fraction = as.numeric(as.character(summary_freqs$species_fraction))

sp_st = reshape2::melt(summary_freqs, "siteID")
sp_st$value = round(sp_st$value, 2)
ggplot(sp_st, aes(fill=variable, y=value, x=siteID)) + 
  geom_bar(position="dodge", stat="identity") + 
  theme_bw() + theme(axis.text.x = element_text(angle=45, hjust=1, vjust=1)) 




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
names(microF1_site) = c("BART", "HARV", "SCBI", "SERC", "DSNY", "OSBS", "GUAN", "STEI", 
                           "KONZ", "UKFS", "GRSM", "MLBS", "DELA", "LENO", "CLBJ", "NIWO", "SRER", "ABBY", "SOAP", "TEAK",
                           "BONA", "DEJU", "HEAL")

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






microF1_site = microF1_site[order(names(microF1_site))]
sp_per_site = sp_per_site[order(names(sp_per_site))]
entries_per_site = entries_per_site[order(names(entries_per_site))]
taxa = names(entries_per_site)
sp_res = cbind.data.frame(microF1_site, sp_per_site, entries_per_site, taxa)
colnames(sp_res) = c("accuracy", "alpha", "effort", "siteID")
library(randomcoloR)
rcol = distinctColorPalette(k = nrow(sp_res), altCol = FALSE, runTsne = FALSE)
ggplot(sp_res, aes(y = accuracy, x =  alpha) ) + geom_point(aes(color = siteID)) + 
  theme_bw() +
  scale_color_manual(values = rcol)+ ylim(0,1)+  xlim(0,18)+
  geom_smooth(method = "gam", formula = y ~ log(x)) + geom_text(aes(label=siteID), position=position_jitter()) + 
  geom_hline(yintercept = 0.76)


names(microF1_dom) <- c("D01", "D02", "D03", "D04", "D05", 
                          "D06", "D07", "D08", "D11", "D13","D14", "D16", "D17", "D19")
microDom = cbind.data.frame(microF1_dom, names(microF1_dom))
colnames(microDom) = c("accuracy", "domainID")
microF1_dom = microF1_dom[order(names(microF1_dom))]

ggplot(microDom, aes(y = accuracy, x = factor(domainID))) + 
  geom_bar(stat="identity",aes(fill = domainID)) +  
  theme_bw() + theme(axis.text.x = element_text(size=14, angle=45, hjust=1, vjust=1)) +
  scale_fill_manual(values = rcol)+ ylim(0,1)+
  geom_hline(yintercept = 0.76)


sp_per_domain = sp_per_domain[order(names(sp_per_domain))]
entries_per_domain = entries_per_domain[order(names(entries_per_domain))]
taxa = names(entries_per_domain)


sp_res = cbind.data.frame(microF1_site, sp_per_site, entries_per_site, taxa)
colnames(sp_res) = c("microF1", "alpha", "effort", "siteID")


p_unc = apply(probabilities[-c(1:3)], 1,  max)
rw = pairs$obs == pairs$pred
p_unc = cbind.data.frame(p_unc, rw)
colnames(p_unc) = c("p(x)", "detected")
p_unc = p_unc[order(p_unc$`p(x)`),]
p_unc["id"] = 1: nrow(p_unc)
ggplot(p_unc, aes(x = detected, y = `p(x)`, fill = detected))+geom_boxplot() + theme_bw()

dm_dt = pairs
dm_dt$obs = factor(dm_dt$obs, levels = unique(data$taxonID))
dm_dt$pred = factor(dm_dt$pred, levels = unique(data$taxonID))
cmdm = confusionMatrix(dm_dt$obs, dm_dt$pred)

f1_by_class = data.frame(cmdm$byClass)
#ggplot(f1_by_class, aes(x = detected, y = `p(x)`, fill = detected))+geom_point() + theme_bw()
f1_by_class$species = rownames(f1_by_class)
ggplot(f1_by_class, aes(y = F1, x = species, color=Prevalence)) + geom_point()+ #scale_color_viridis_c()+
  theme_bw() + theme(axis.text.x = element_text(size=14, angle=45, hjust=1, vjust=1)) + ylim(0,1)+coord_flip()
#precision
ggplot(f1_by_class, aes(y = Precision, x = species, color=Prevalence)) + geom_point()+ #scale_color_viridis_c()+
  theme_bw() + theme(axis.text.x = element_text(size=14, angle=45, hjust=1, vjust=1)) + ylim(0,1)+coord_flip()
#recall
ggplot(f1_by_class, aes(y = Recall, x = species, color=Prevalence)) + geom_point()+ #scale_color_viridis_c()+
  theme_bw() + theme(axis.text.x = element_text(size=14, angle=45, hjust=1, vjust=1)) + ylim(0,1)+coord_flip()+
  geom_hline(yintercept = 0.76)


bad_sp = f1_by_class %>% filter(F1 < 0.4)
cmdm$table[,rownames(cmdm$table) == "CATO6"]
cmdm$table[rownames(cmdm$table) == "CATO6",] #LIST

hm <- cmdm$table %>% data.frame
hm <- hm %>%
  mutate(Prediction = factor(Prediction, levels = sort(unique(as.character(Reference)))), # alphabetical order by default
         Reference = factor(Reference, levels = rev(sort(unique(as.character(Reference)), decreasing = T)))) # force reverse alphabetical order

ggplot(hm, aes(x=Prediction, y=Reference, fill=Freq)) +
  geom_tile() + theme_bw() + coord_equal() +
  theme(axis.text.x = element_text(size=8, angle=45, hjust=1, vjust=1)) +
  theme(axis.text.y = element_text(size=8, angle=45, hjust=1, vjust=1)) +
  scale_fill_distiller(palette="Greens", direction=1) +
  guides(fill=F) + # removing legend for `fill`
  labs(title = "Value distribution") + # using a title instead
  geom_text(data=subset(hm, Freq > 0), aes(label=Freq),size = 3, color="black") # printing values

fam_confu = readr::read_csv("~/Documents/fam_per_confucio.csv")
fam_confu$Predictions = factor(fam_confu$Predictions,
                               levels = sort(unique(fam_confu$Reference)))
fam_confu$Reference = factor(fam_confu$Reference,
                               levels = sort(unique(fam_confu$Reference)))
cmfm =  confusionMatrix(fam_confu$Reference, fam_confu$Predictions)
hm = data.frame(cmfm$table)
ggplot(hm, aes(x=Prediction, y=Reference, fill=Freq)) +
  geom_tile() + theme_bw() + coord_equal() +
  theme(axis.text.x = element_text(size=8, angle=45, hjust=1, vjust=1)) +
  theme(axis.text.y = element_text(size=8, angle=45, hjust=1, vjust=1)) +
  scale_fill_distiller(palette="Greens", direction=1) +
  guides(fill=F) + # removing legend for `fill`
  labs(title = "Value distribution") + # using a title instead
  geom_text(data=subset(hm, Freq > 0), aes(label=Freq),size = 3, color="black") # printing values

#QUERC -within genus
#CELA - LIST
#CATO ` frax`