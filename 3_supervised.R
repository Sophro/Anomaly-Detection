# July 2019
# supervised with RF, XGB and down, up and smote sampling 

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
folder_out <- "data/dat_from_RF/"
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


# reduced features_names removing correlated variables: out_gini, net_value, num_addresses, ratio_deg --> 13 features
selected_features <- c("type", "in_sum", "out_mean", "in_mean", "lifetime",  "delay_min", "delay_max", "act_day",
                       "max_tx_per_day","out_gini", "out_degree", "ratio_degree", "in_degree", "num_addresses")

selected_features_old <- c("type", "out_sum", "in_sum", "out_mean", "in_mean", "lifetime", "delay_mean", "delay_min", "delay_max", "act_day",
                       "max_tx_per_day","in_gini", "out_degree", "in_degree")

# only for check
# selected_features_min <- c("type", "out_sum", "in_sum")

# Try different classifier models
#
# basic Random Forest 
# Metric= By default, possible values are "RMSE" and "Rsquared" for regression and "Accuracy" and "Kappa" for classification.
set.seed(37756)
param_cv <- trainControl(method = "repeatedcv",
                         classProbs = TRUE,
                         number = 10, 
                         repeats = 2)

mod_rf_orig <- train(as.factor(type) ~ . , 
                     data= subset(training, select = selected_features), 
                     method="rf",
                     metric =   "Kappa", # "Accuracy" default = accuracy
                     trControl=param_cv)
cM_rf_orig   <- confusionMatrix(predict(mod_rf_orig, subset(testing, select = selected_features)), as.factor(testing$type))
save(mod_rf_orig, file = "mod_rf_orig.Rdata")

# Build down-sampled model
param_cv$sampling <- "down"
param_cv$seeds <- mod_rf_orig$control$seeds # use exactly the same seeds
down_mod <- train(type ~ .,
                  data = subset(training, select = selected_features),
                  method = "rf",
                  verbose = FALSE,
                  metric = "Kappa",
                  trControl = param_cv)
print(down_mod$finalModel) # to compare with orig
cM_down_mod   <- confusionMatrix(predict(down_mod, subset(testing, select = selected_features)), as.factor(testing$type))
save(down_mod, file = "down_mod.Rdata")

# Build up-sampled model
param_cv$sampling <- "up"
param_cv$seeds <- mod_rf_orig$control$seeds # use exactly the same seeds
up_mod <- train(type ~ .,
                  data = subset(training, select = selected_features),
                  method = "rf",
                  verbose = FALSE,
                  metric = "Kappa",
                  trControl = param_cv)
print(up_mod$finalModel) # to compare with orig
cM_up_mod   <- confusionMatrix(predict(up_mod, subset(testing, select = selected_features)), as.factor(testing$type))
save(up_mod, file = "up_mod.Rdata")

#install.packages("ranger")
library("ranger")
my_tune_grid <- expand.grid(
  .mtry = 2:13,
  .splitrule = "gini",
  .min.node.size = c(10, 20)
)
ranger_mod <- train(type  ~ ., data = subset(training, select = selected_features),
                     method = "ranger",
                     trControl = trainControl(method="cv", number = 5, verboseIter = T, classProbs = T),
                     tuneGrid = my_tune_grid,
                     num.trees = 100,
                     importance = "permutation")
print(ranger_mod$finalModel)
cM_ranger_mod   <- confusionMatrix(predict(ranger_mod, subset(testing, select = selected_features)), as.factor(testing$type))
save(ranger_mod, file = "ranger_mod.Rdata")

# Build SMOTE model
param_cv$sampling <- "smote"
# param_cv$seeds <- ranger_mod$control$seeds # use exactly the same seeds as ranger NOT RF
smote_mod <- train(type ~ .,
                data = subset(training, select = selected_features),
                method = "ranger",
                verbose = FALSE,
                metric = "Kappa",
#                trControl = param_cv,
#                tuneGrid = my_tune_grid,
                maximize = TRUE)
print(smote_mod$finalModel) # to compare with orig
cM_smote_mod   <- confusionMatrix(predict(smote_mod, subset(testing, select = selected_features)), as.factor(testing$type))
save(smote_mod, file = "smote_mod.Rdata")

#  XGB  with best yperparameters (from script 7_Tune_model_xgboost.R)
set.seed(37756)
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
print(xgb_best$finalModel) # to compare with orig
save(xgb_best, file = "xgb_best.Rdata")
cM_xgb_best   <- confusionMatrix(predict(xgb_best, subset(testing, select = selected_features)), as.factor(testing$type))

############### end of models

# Start comparison
model_list <- list(original = mod_rf_orig,
                   down = down_mod,
                   up = up_mod,
                   ranger = ranger_mod,
                   smote = smote_mod,
                   xgb_best)

models_compare <- resamples(list(RF_orig = mod_rf_orig,  up_mod = up_mod, down_mod = down_mod, xgb_best = xgb_best)) 
# ranger = ranger_mod, smote = smote_mod))  There are different numbers of resamples using also these two  models
summary(models_compare)                                 

dev.new()
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)


# final metrics
model_for_prediction <-  xgb_best  # smote_mod  #ranger_mod #  smote_mod # xgb_best # mod_rf_orig # down_mod # up_mod #
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

# print final results
anomaly_score <- data.frame(rbind(a1,a2,a3,a4))
row.names(anomaly_score) <- c("Ponzi", "hacks", "LSeed", "realrandom")
colnames(anomaly_score) <- c("Balanced_Accuracy", "Precision", "Sensitivity", "F1")
print(signif(anomaly_score,3))
cat("MCC =", MCC,"\n")
print(final_CM$table)
