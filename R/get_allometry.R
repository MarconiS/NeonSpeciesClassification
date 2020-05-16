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


allometry_dataset = allometry_dataset %>% filter(height > 3) %>%
  filter(stemDiameter < 150) %>%
  filter(height < 55)

allometry_forested = allometry_dataset %>%
  filter(!nlcdClass %in% c("emergentHerbaceousWetlands", "cultivatedCrops", "grasslandHerbaceous", "pastureHay"))

#allometry_forested$stemDiameter = log(allometry_forested$stemDiameter)
#split train test
train = allometry_forested %>%
  group_by(nlcdClass, taxonID, siteID) %>%
  sample_frac(0.8)
test = allometry_forested %>% filter(!individualID %in% train$individualID)
library(brms)
#run model
fit <- brm(height ~ s(stemDiameter) + (1| s | siteID) + (1 | t | taxonID),  
           data = train,
           cores = 2,
           seed = 12,
           family = lognormal(),
           control = list(adapt_delta = 0.99),
           thin = 10, #refresh = 0,
           prior = set_prior(horseshoe(df = 3)),
           chains = 2,
           iter = 3000)

test_r2 = bayes_R2(fit, newdata = test)
#
print(test_r2)
prds = predict(fit, newdata = test)
prds = cbind.data.frame(test, prds)
saveRDS(list(fit,prds), "./lognormal.rds")

ggplot(allometry_forested, aes(x = (stemDiameter), y = (height), color = siteID)) + geom_point() + theme_minimal()





