# July 2019  reads from output Script1 
#  clean
rm(list=ls())
suppressPackageStartupMessages({
  library("purrr")
  library("readr")
  library("dplyr")
  library("tidyr")
  library("reldist")
  library("caret")
  library("robustbase")
  library("rrcov")
  library("randomForest")
})
set.seed(37756)

# Folder names:
folder_in <- "data"

dat_inp <- as.data.frame(
  read_delim(file.path(folder_in,
                       paste0("address_features.csv.gz")),
             delim = ";", col_names = TRUE))

dat <- dat_inp %>% filter(!is.na(out_degree)) 
dat <- dat %>% filter(type != "RWare")
dat <- dat %>% filter(type != "random")

subset( dat, select = -c(address)) %>% group_by(type) %>% count()
subset( dat, select = -c(address)) %>% group_by(type) %>% summarize_all(mean)


# principal component analysis
# center and scale data first 
xx <- as.data.frame(scale(subset(dat, select = -c(type, address, in_sd, out_sd, lifetime_days)), center =TRUE, scale =TRUE)) # 
# xx <- as.data.frame(scale(subset(dat, select = -c(type, address)), center =TRUE, scale =TRUE)) # 
Principal_comp <- prcomp(xx) 
summary(Principal_comp)
# from summary the first 6 components achieve almost 90% of the variance

plot(Principal_comp, type="l")

# sort only PC1 in decreasing order
sort(Principal_comp$rotation[,1], decreasing = TRUE)
# most important in first component are: lifetime, delay_max,delay_mean,delay_min, act_day, in_degree, in_gini

dev.new()
biplot(Principal_comp) # highly correlated out_sum & out_mean ; in_sum & in_mean ; delay_min & delay_mean---> remove out_sum, in_sum and in_mean 


# repeat removing  in_sum, in_mean, delay_min
xx <- as.data.frame(scale(subset(dat, select = -c(type, address, in_sd, out_sd, lifetime_days, in_sum, in_mean, delay_min )), center =TRUE, scale =TRUE))
Principal_comp <- prcomp(xx) 
summary(Principal_comp)
sort(Principal_comp$rotation[,1], decreasing = TRUE)
plot(Principal_comp, type="l")
# the first 6 components achieve about 90% of the variance
# most important in first component are: delay_max,delay_mean,delay_min, act_day, in_degree, in_gini
dev.new()
biplot(Principal_comp) # highly correlated out_sum & in_sum ; out_mean & in_mean remove in_sum and in_mean 

# check correlations
most_important <- subset(dat, select = -c(type, address, in_sd, out_sd, lifetime_days, in_sum, in_mean, delay_min ))
panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r)
}
dev.new()
pairs(most_important,lower.panel=panel.smooth,upper.panel=panel.cor)
# correlation for ech pairs
cor(most_important)
# hightly correlated: out_mean out_sum; out_sum out_mean ; in_gini out_gini ; ratio_deg net_value

# create train and testing sets
partition = 0.6 # = percentage which goes to training
inTrain <- createDataPartition(y= dat$type, p=partition, list=FALSE) 
training <- dat[ inTrain,] 
testing  <- dat[-inTrain,]
dim(training)
dim(testing)
prop.table(table(training$type))

# features_names <- paste(colnames(dat), collapse =",")# not OK 
all_features <- c("type","out_sum", "in_sum", "out_mean", "in_mean", "lifetime", "delay_mean", "delay_min", "delay_max", "act_day",
                  "max_tx_per_day","in_gini", "out_gini", "out_degree", "in_degree", "num_addresses", "net_value", "ratio_degree") #, "out_sd", "in_sd")

# reverse features
rev_features <- rev(all_features)
# or use a different order
rev_features <- c("type" , "ratio_degree",  "net_value" ,     "num_addresses",  "in_degree",      "out_degree",     "out_gini" ,      "in_gini",       
                  "max_tx_per_day", "act_day",        "delay_max" ,     "delay_min" ,     "delay_mean",     "lifetime" ,      "in_mean",       
                  "out_mean",       "in_sum" ,        "out_sum" )

# reduced features_names removing correlated variables: --> 13 features
# in decreasing importance order
selected_features <- c("type", "in_sum", "out_mean", "in_mean", "lifetime",  "delay_min", "delay_max", "act_day",
                       "max_tx_per_day","out_gini", "out_degree", "ratio_degree", "in_degree", "num_addresses")
selected_features_10 <- c("type", "out_sum", "lifetime", "delay_min", "delay_max", "act_day",
                         "max_tx_per_day","in_gini", "out_degree", "in_degree")

#selected_features_3 <- c("type", "out_sum", "in_sum")

use_these_features <- selected_features_10 # selected_features #all_features # rev_features 


# repeat PCA with selected_features
xx <- as.data.frame(scale(subset(dat, select = c("in_sum", "out_mean", "in_mean", "lifetime",  "delay_min", "delay_max", "act_day",
                                                  "max_tx_per_day","out_gini", "out_degree", "ratio_degree", "in_degree", "num_addresses")), center =TRUE, scale =TRUE))
Principal_comp <- prcomp(xx) 
summary(Principal_comp)
sort(Principal_comp$rotation[,1], decreasing = TRUE)
plot(Principal_comp, type="l")
# the first 6 components achieve about 90% of the variance
# most important in first component are: delay_max,delay_mean,delay_min, act_day, in_degree, in_gini
dev.new()
biplot(Principal_comp) # highly correlated out_sum & in_sum ; out_mean & in_mean remove in_sum and in_mean 


# check importances:
rfm <- randomForest(as.factor(type) ~ ., subset(training, select = selected_features))
dev.new()
varImpPlot(rfm, main = "importances")


# k-fold (k = 10 repeated 2 times) with caret
param_cv <- trainControl(method = "repeatedcv", 
                         classProbs = TRUE,
                         number = 10, 
                         repeats = 2) 
#
# calculate RF model with increasing number of features from 1 to length of selected features
# Metric= possible values are "RMSE" and "Rsquared" for regression and "Accuracy" and "Kappa" for classification.
for (nn in seq(1:(length(use_these_features)-1))) {
print(nn)
mod_rf_fold <- train(as.formula(paste("as.factor(type) ~ ", paste(head(use_these_features[-1], nn), collapse=" + "))),  # remove type !! 
                     data=subset(training, select = use_these_features), 
                     method="rf", 
                     metric =   "Kappa", # "Accuracy" default = accuracy
                     trControl=param_cv)

importance <- varImp(mod_rf_fold, scale =FALSE)
print(importance)

cM1   <- confusionMatrix(predict(mod_rf_fold, subset(testing, select = use_these_features)), as.factor(testing$type))

final_CM <- cM1
# write final output in a data frame

anomaly_score <- data.frame()

a1 <- data.frame(cbind(
                    nrow(importance$importance),
                    final_CM$byClass["Class: Ponzi","Balanced Accuracy"], 
                    final_CM$byClass["Class: Ponzi","Precision"], 
                    final_CM$byClass["Class: Ponzi","Sensitivity"], 
                    final_CM$byClass["Class: Ponzi","F1"] 
                     ))
a2 <- data.frame(cbind(
                    nrow(importance$importance),
                    final_CM$byClass["Class: hacks","Balanced Accuracy"], 
                    final_CM$byClass["Class: hacks","Precision"], 
                    final_CM$byClass["Class: hacks","Sensitivity"], 
                    final_CM$byClass["Class: hacks","F1"] 
                     ))

a3 <- data.frame(cbind(
                    nrow(importance$importance),
                    final_CM$byClass["Class: LSeed","Balanced Accuracy"], 
                    final_CM$byClass["Class: LSeed","Precision"], 
                    final_CM$byClass["Class: LSeed","Sensitivity"], 
                    final_CM$byClass["Class: LSeed","F1"] 
                     ))
anomaly_score <- data.frame(rbind(a1,a2,a3))
colnames(anomaly_score) <- c("N.Features", "Balanced_Accuracy", "Precision", "Sensitivity", "F1")
row.names(anomaly_score) <- c("Ponzi", "hacks", "LSeed")
print(signif(anomaly_score,3))

#write.table(anomaly_score, file = "./from_reduce_features_v4.csv",  sep = ",", col.names = NA) # col.names =NA insert a blanck space on first column
write.table(anomaly_score, file = "./from_reduce_features_v4.csv",  sep = ",", col.names = FALSE, append=TRUE)

#write.table(anomaly_score, file = "./from_reduce_features_v4_all.csv",  sep = ",", col.names = NA)
write.table(anomaly_score, file = "./from_reduce_features_v4_all.csv",  sep = ",", col.names = TRUE, append=TRUE)
write.table(paste(mod_rf_fold$coefnames), file = "./from_reduce_features_v4_all.csv",  sep = ",", append=TRUE)

print(nn)
}

#
# read results
dat_out <- read.csv(file = "./from_reduce_features_v4.csv", header= TRUE)


