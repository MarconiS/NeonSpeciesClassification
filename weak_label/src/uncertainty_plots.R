#evaluate uncertainty
wrong_ids = pairs$individualID[!(pairs$obs == pairs$pred)]
pdist_wrong = probabilities %>% filter(individualID %in% wrong_ids)

# median distribution for misclassified species by species
dist_sp = pdist_wrong %>% select(-one_of("X1", "individualID")) %>% group_by(taxonID) %>%
  summarize_each(median)
dist_sp = reshape2::melt(dist_sp)
ggplot(dist_sp, aes(x = variable, y = value)) + geom_point() + facet_wrap(.~taxonID)

highish = dist_sp %>% filter(value > 0.01)
highish = highish[!highish$taxonID == highish$variable,]

ggplot(highish, aes(x = as.character(variable), y = value)) + geom_point() + ylim(0,1)+
  facet_wrap(.~taxonID, drop = T, scales = "free_x") + theme_bw() + geom_hline(yintercept = 0.1)


# make a quantitative analysis using ranking
# Can we look at it more continuosly: 0-10 which fraction correctly classified? 
all_pdist = probabilities %>% select(-one_of("X1", "individualID")) 
p_max = apply(all_pdist[,-1], 1, function(x)(max(x)))
who_max = lapply(1:nrow(all_pdist), function(x)(colnames(all_pdist[which(all_pdist[x,]==p_max[x])])))
who_max = unlist(who_max)
final_boxes = cbind.data.frame(all_pdist[["taxonID"]], who_max, p_max)
bin = c(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1)
fraction_ = rep(NA, 10)
for(ii in 2:11){
  ith_class = final_boxes %>% filter(final_boxes$p_max < bin[ii] & 
                                       final_boxes$p_max > bin[ii-1])
  fraction_[ii-1] = sum(ith_class[,1] != ith_class[,2])/nrow(ith_class)
}

uncertainty_curve = cbind.data.frame(bin[-11], fraction_)
colnames(uncertainty_curve) = c("majority_p", "fraction_misclassified")
ggplot(uncertainty_curve, aes(x = majority_p, y = fraction_misclassified)) + ylim(0,1) + xlim(0,1)+
  geom_point() + theme_bw() + geom_abline(intercept = 1, slope = -1) + stat_smooth(method="lm", se=FALSE)
