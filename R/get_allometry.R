#create allometric model to infer height from dbh, species, site and nlcd
library(tidyverse)
vst <- readr::read_csv("./weak_label/indir/csv/vst_field_data.csv")

#get only variables of interest to build dbh-h allometry
allometry_dataset = vst %>% select(individualID, height, stemDiameter, 
                                   nlcdClass, taxonID, siteID, eventID) 
allometry_dataset = allometry_dataset[complete.cases(allometry_dataset),]
#sample each tree only once
allometry_dataset = allometry_dataset %>% group_by(individualID) %>%
  sample_n(1)

#split train test
train = allometry_dataset %>%
  group_by(nlcdClass, taxonID, siteID) %>%
  sample_frac(0.8)
test = allometry_dataset %>% filter(!individualID %in% train$individualID)
library(brms)
#run model
fit <- brm(height ~ stemDiameter + (1| siteID/nlcdClass) + (1| taxonID),  
           data = train,
           cores = 2,
           seed = 12,
           family = lognormal(),
           control = list(adapt_delta = 0.99),
           thin = 10, #refresh = 0,
           #prior = set_prior(horseshoe(df = 3)),
           chains = 2,
           iter = 3000)
