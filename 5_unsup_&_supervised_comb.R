# 18.05.2019 updated June/July 2019
# unsupervised + supervised combined
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
  library(lime)
  library(ineq)
  library("mltools")
})
#set.seed(502973)
#set.seed(37756)
#set.seed(205730935)
#set.seed(0967323)
set.seed(123)
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


# plot all data 
dev.new()
gg <- ggplot(dat, aes(x=(in_gini), y=(in_sum), colour=type)) + geom_point() +  scale_y_log10() + 
  scale_colour_manual(values = c( "Red", "Green","Blue", "Gray")) + ggtitle("Full data set by type") 
plot(gg)


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


# reduced features removing correlated variables: out_gini, net_value, num_addresses, ratio_deg --> 13 features
selected_features <- c("type", "in_sum", "out_mean", "in_mean", "lifetime",  "delay_min", "delay_max", "act_day",
                       "max_tx_per_day","out_gini", "out_degree", "ratio_degree", "in_degree", "num_addresses")
#
selected_features_plus_address <- c("address","type", "in_sum", "out_mean", "in_mean", "lifetime",  "delay_min", "delay_max", "act_day",
                       "max_tx_per_day","out_gini", "out_degree", "ratio_degree", "in_degree", "num_addresses")


# reduced features removing correlated variables: out_gini, net_value, num_addresses, ratio_deg --> 13 features
selected_features <- c("type", "in_sum", "out_mean", "in_mean", "lifetime",  "delay_min", "delay_max", "act_day",
                       "max_tx_per_day","out_gini", "out_degree", "ratio_degree", "in_degree")
selected_features_plus_address <- c("address","type", "in_sum", "out_mean", "in_mean", "lifetime",  "delay_min", "delay_max", "act_day",
                                    "max_tx_per_day","out_gini", "out_degree", "ratio_degree", "in_degree")


# selected_features_old <- c("type", "out_sum", "in_sum", "out_mean", "in_mean", "lifetime", "delay_mean", "delay_min", "delay_max", "act_day",
#                            "max_tx_per_day","in_gini", "out_degree", "in_degree")

#selected_features <- c("type", "out_sum", "in_sum")


xm <- subset(testing, select = selected_features) # BEWARE !! can be dat, training or testing
xm <- subset(xm, select = -c(type)) 
xm <- as.data.frame(scale(xm, center =TRUE, scale =TRUE))
# sum(is.na(dat$in_gini)) 

# mahalanobis distance
# change number of outliers to find
kk = 40
# xxxx <- mahalanobis(xm, colMeans(xm), CovMcd(xm)@cov)
xxxx <- mahalanobis(xm, colMeans(xm), cov(xm))
sorted_mahal <- sort(xxxx, decr=T)  # by values
order_mahal <- order(xxxx, decreasing =T)  # by lines 
cutoff = sort(xxxx, decr=T)[kk] # cutoff on the VALUE (not on the number of outliers)
xm$outliers <- factor(ifelse(xxxx >= cutoff, "TRUE", "FALSE"))
xm$maha_score <- xxxx
prop.table(table(xm$outliers))

# final data frame with outliers from Mahalanobis distance
# xm <- xm[order(-xm$maha_score),] # descending order
 
# check only outliers
#maha_outliers <- xm[xm$outliers == TRUE,]


# supervised method (use default or choose the best from make_all2)

################################   default model using random Forest. Very fast and not so bad
# ntree: Number of trees to grow (i.e. nrow(dat))
# mtry =  No. of variables tried at each split: by default = srqt(ncol(dat))
rfm <-  randomForest(as.factor(type) ~ ., subset(training, select = selected_features)) 

dev.new()
varImpPlot(rfm, main = "importances")

# make predictions 
cM0 <- confusionMatrix(predict(rfm , subset(testing, select = selected_features)), as.factor(testing$type))
MCC0 <- mcc(predict(rfm, subset(testing, select = selected_features)), as.factor(testing$type))
cat("MCC0 =", MCC0)
############################################# end default model using random Forest

#  XGB Extreme Gradient Boosting   with best yperparameters (from script 7_Tune_model_xgboost.R)
#set.seed(502973)
#set.seed(37756)
#set.seed(205730935)
#set.seed(0967323)
set.seed(123)
train_xgb_best <- trainControl(method = "repeatedcv",
                               number = 10, 
                               repeats = 2,
                               verboseIter = FALSE,
                               classProbs = TRUE,
                               allowParallel = FALSE # FALSE for reproducible results 
)
grid_best <- expand.grid(
  nrounds = 300,  # 150 used in all models by default 
  max_depth = 4,  # 3
  eta = 0.1,      # 0.4
  gamma = 0,      # 0
  colsample_bytree = 1, # 0.6 
  min_child_weight = 1, # 1
  subsample = 1         # 1 
)

xgb_best <- train(
  data = subset(training, select = selected_features),
  as.factor(type) ~ .,
  trControl = train_xgb_best,
  tuneGrid = grid_best,
  method = "xgbTree",
  metric = "Kappa", # default accuracy 
  verbose = TRUE
)

save(xgb_best, file = "xgb_best.Rdata")
cM_xgb_best   <- confusionMatrix(predict(xgb_best, subset(testing, select = selected_features)), as.factor(testing$type))

# choose which model for predictions and final CM
model_for_prediction <- xgb_best # mod_rf_fold 
cM_best <- confusionMatrix(predict(model_for_prediction, subset(testing, select = selected_features)), as.factor(testing$type))
print(cM_best$table)

# features importance 
varImp(model_for_prediction)

# save testing and final outcomes in x_final 
x_final <- subset(testing, select = selected_features_plus_address)
x_final$predicted <- predict(model_for_prediction, subset(testing, select = selected_features))
x_final$maha_score <- xm$maha_score
x_final$outliers <- xm$outliers
x_final_decreasing  <- x_final[order(-x_final$maha_score),] # descending order

check_Mahal_outliers <- count(x_final[x_final$outliers == TRUE,])
print(check_Mahal_outliers)

################### check final summary 

# final summary by type
type_mean <- select(x_final, -c(address, starts_with("predicted"), outliers)) %>% group_by(type) %>% summarise_all(mean)
type_sd <-select(x_final, -c(address, starts_with("predicted"), outliers)) %>% group_by(type) %>% summarise_all(sd)
type_Gini <-select(x_final, -c(address, starts_with("predicted"), outliers)) %>% group_by(type) %>% summarise_all(Gini)
type_entropy <-select(x_final, -c(address, starts_with("predicted"), outliers)) %>% group_by(type) %>% summarise_all(entropy)
prop.table(table(x_final$type))

# final summary by outliers (from Mahal)
outliers_mean <- select(x_final, -c(address, starts_with("predicted"), type)) %>% group_by(outliers) %>% summarise_all(mean)
outliers_sd <- select(x_final, -c(address, starts_with("predicted"), type)) %>% group_by(outliers) %>% summarise_all(sd)
outliers_Gini <- select(x_final, -c(address, starts_with("predicted"), type)) %>% group_by(outliers) %>% summarise_all(Gini)
outliers_entropy <- select(x_final, -c(address, starts_with("predicted"), type)) %>% group_by(outliers) %>% summarise_all(entropy)
prop.table(table(x_final$outliers))

# final summary by predicted
predicted_mean <- select(x_final, -c(address, outliers, type)) %>% group_by(predicted) %>% summarise_all(mean)
predicted_sd <- select(x_final, -c(address, outliers, type)) %>% group_by(predicted) %>% summarise_all(sd)
predicted_Gini <- select(x_final, -c(address, outliers, type)) %>% group_by(predicted) %>% summarise_all(Gini)
predicted_entropy <- select(x_final, -c(address, outliers, type)) %>% group_by(predicted) %>% summarise_all(entropy)
prop.table(table(x_final$predicted))

# final plots
dev.new()
gg <- ggplot(x_final, aes(x=abs(out_gini), y=(in_sum), colour=type)) + geom_point() + scale_y_continuous(trans='log10', limits =c(1.e-2, 1e6)) + 
      scale_colour_manual(values = c("Red", "Green","Blue", "Gray")) + ggtitle("type from testing set") 
plot(gg)
#dev.copy2pdf(out.type = "pdf", file= "True.pdf")
#dev.off() 


dev.new()
gg <- ggplot(x_final, aes(x=abs(out_gini), y=(in_sum), colour=outliers)) + geom_point()  + scale_y_continuous(trans='log10' , limits =c(1.e-2, 1e6)) + 
  scale_colour_manual(values = c("gray","red")) + ggtitle("outliers from Mahalanobis on testing set") # +
# geom_point(aes(x=0.43208303, y=1.449008e+02),color= "Blue")
plot(gg)
#dev.copy2pdf(out.type = "pdf", file= "Outliers.pdf")
#dev.off() 


dev.new()
gg <- ggplot(x_final, aes(x=abs(out_gini), y=(in_sum), colour=predicted)) + geom_point() + scale_y_continuous(trans='log10', limits =c(1.e-2, 1e6)) + 
  scale_colour_manual(values = c("Red", "Green","Blue", "Gray")) + ggtitle("Predicted with Best model on testing set") 
plot(gg)
#dev.copy2pdf(out.type = "pdf", file= "Predicted.pdf")
#dev.off() 


write.csv(type_mean, file = "./type_mean.csv")

# in particular for LSeed
# type
dev.new()
gg <- ggplot(x_final, aes(x=(delay_min), y=(lifetime), colour=type)) + geom_point() + scale_y_log10() + scale_x_log10() +
  scale_colour_manual(values = c("Red", "Green","Blue", "Gray")) + ggtitle("type from testing set") 
plot(gg)
#predicted
dev.new()
gg <- ggplot(x_final, aes(x=(delay_min), y=(lifetime), colour=predicted)) + geom_point() + scale_y_log10() + scale_x_log10() +
  scale_colour_manual(values = c("Red", "Green","Blue", "Gray")) + ggtitle("Predicted with Best model on testing set") 
#  geom_errorbar(aes(ymin=predicted_mean$lifetime, ))
plot(gg)

dev.new()
gg <- ggplot(x_final, aes(x=(delay_min), y=(lifetime), colour=outliers)) + geom_point() + scale_y_log10() + scale_x_log10() +
  scale_colour_manual(values = c( "Gray", "Red")) + ggtitle("Predicted with Best model on testing set") 
#  geom_errorbar(aes(ymin=predicted_mean$lifetime, ))
plot(gg)


##################################################################

# final results for tables
final_results <- x_final_decreasing %>% select(type, address, predicted, outliers, maha_score)
final_results_short <- final_results %>% mutate(address = substr(address,1,10), maha_score = round(final_results$maha_score, digits=2)) 


# check how many anomalies are detected (depends if Ponzi & hacks or also include LSeed)
# considering anomaly only Ponzi & hacks
final_outliers             <- final_results %>% mutate(is_anomaly = factor(ifelse((type == "hacks" | type == "Ponzi"), "TRUE", "FALSE"))) %>% 
                              mutate(anom_eq_outlier = factor(ifelse(((is_anomaly == "TRUE" & outliers == "TRUE") | (is_anomaly == "FALSE" & outliers == "FALSE")), "TRUE", "FALSE")))
MCC_unsup <- mcc(as.factor(final_outliers$outliers), as.factor(final_outliers$is_anomaly))


# considering anomaly Ponzi, hacks & LSeed
final_outliers_with_LSeed  <- final_results %>% mutate(is_anomaly = factor(ifelse((type == "hacks" | type == "Ponzi" | type == "LSeed"), "TRUE", "FALSE"))) %>% 
                              mutate(anom_eq_outlier = factor(ifelse(((is_anomaly == "TRUE" & outliers == "TRUE") | (is_anomaly == "FALSE" & outliers == "FALSE")), "TRUE", "FALSE")))
MCC_unsup_with_LSeed <- mcc(as.factor(final_outliers_with_LSeed$outliers), as.factor(final_outliers_with_LSeed$is_anomaly))

# for supervised MCC is the same 
MCC_sup <- mcc(as.factor(final_outliers$predicted), as.factor(final_outliers$type))

print(cbind(MCC_unsup, MCC_unsup_with_LSeed,MCC_sup))


suspicious   <- final_results %>% filter(outliers == TRUE)
likely  <- final_results %>% filter(outliers == TRUE & c(predicted == "hacks" | predicted == "Ponzi" | predicted == "LSeed") )

dev.new()
gg <- ggplot(x_final, aes(x=(in_gini), y=(in_sum), colour=outliers)) + geom_point()  + scale_y_continuous(trans='log10' , limits =c(1.e-2, 1e6)) + 
  scale_colour_manual(values = c("gray","red")) + ggtitle("outliers from Mahalanobis on testing set") # +
# geom_point(aes(x=0.43208303, y=1.449008e+02),color= "Blue")
plot(gg)


# check final results with lime 
# final_for_lime <- x_final  %>% filter(type == "realrandom" & predicted == "hacks") # predicted_xgb # check if false positive 
final_for_lime <- x_final  %>% filter( predicted  == "Ponzi") %>% sample_n(6, replace =FALSE)# predicted_xgb # check if predict hacks is robust
# final_for_lime <- x_final %>% filter( predicted == "LSeed") %>% sample_n(6, replace =FALSE)# predicted_xgb # check if predict LSeed is robust, select only 6
#final_for_lime <- sample_n(x_final, 6, replace =FALSE) # select only 6 , will change for each new model # check randomly 
explainer_caret <- lime(subset(final_for_lime, select = -c(type, address)), model_for_prediction, n_bins = 4)
explanation <- lime::explain(subset(final_for_lime, select = -c(type, address)), explainer_caret, n_labels = 1, n_features = 4, feature_select = "highest_weights")
explanation[, 2:9]
dev.new()
plot_features(explanation, ncol = 2)
dev.new()
plot_explanations(explanation)
#dev.copy2pdf(out.type = "pdf", file= "plot_lime_hacks.pdf")
#dev.off() 

# Print tables in latex
library(huxtable)
#hux(predicted_mean, add_colnames = TRUE)
#hux(final_results, add_colnames = TRUE)
hux(final_results_short, add_colnames = TRUE)

library(xtable)
#xtable(s_mean)
#print(xtable(predicted_mean, caption = "average values for each predicted anomaly....change digits"))
xtable(final_results_short[1:10,], add_colnames = TRUE)
xtable(cM_xgb_best$table, add_colnames = TRUE)


# final metrics for the selected model BEWARE: choose the best one!
model_for_prediction <- xgb_best # rfm #  mod_rf_fold # xgb_best
cM_best <- confusionMatrix(predict(model_for_prediction, subset(testing, select = selected_features)), as.factor(testing$type))
final_CM <- cM_best
MCC <- mcc(predict(model_for_prediction, subset(testing, select = selected_features)), as.factor(testing$type))
cat("MCC =", MCC)

anomaly_score <- data.frame()

a1 <- data.frame(cbind(
  final_CM$byClass["Class: Ponzi","Balanced Accuracy"], 
  final_CM$byClass["Class: Ponzi","Precision"], 
  final_CM$byClass["Class: Ponzi","Sensitivity"], 
  final_CM$byClass["Class: Ponzi","F1"] 
))
a2 <- data.frame(cbind(
  final_CM$byClass["Class: hacks","Balanced Accuracy"], 
  final_CM$byClass["Class: hacks","Precision"], 
  final_CM$byClass["Class: hacks","Sensitivity"], 
  final_CM$byClass["Class: hacks","F1"] 
))
a3 <- data.frame(cbind(
  final_CM$byClass["Class: LSeed","Balanced Accuracy"], 
  final_CM$byClass["Class: LSeed","Precision"], 
  final_CM$byClass["Class: LSeed","Sensitivity"], 
  final_CM$byClass["Class: LSeed","F1"] 
))
a4 <- data.frame(cbind(
  final_CM$byClass["Class: realrandom","Balanced Accuracy"], 
  final_CM$byClass["Class: realrandom","Precision"], 
  final_CM$byClass["Class: realrandom","Sensitivity"], 
  final_CM$byClass["Class: realrandom","F1"] 
))


# print final metrics
anomaly_score <- data.frame(rbind(a1,a2,a3,a4))
row.names(anomaly_score) <- c("Ponzi", "hacks", "LSeed", "realrandom")
colnames(anomaly_score) <- c("Balanced_Accuracy", "Precision", "Sensitivity", "F1")
print(signif(anomaly_score,3))
cat("MCC =", MCC)
print(final_CM$table)

#
testing %>% group_by(type) %>% count()
prop.table(table(testing$type))

