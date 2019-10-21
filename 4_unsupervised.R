# February 18th 2019
# unsupervised methods
#  clean
rm(list=ls())
#set.seed(394)
set.seed(37756)
suppressPackageStartupMessages({
  library("purrr")
  library("readr")
  library("dplyr")
  library("tidyr")
  library("reldist")
  library("ggplot2")
  library("caret")

})

set.seed(37756)

# Folder names:
#folder_in  <- "C:/Users/rollets/Desktop/Anomalies/dat_for_an/"
folder_out <- "C:/Users/rollets/Desktop/Anomalies/dat_from_RF/"
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
                       "max_tx_per_day","in_gini", "out_degree", "ratio_degree", "in_degree", "num_addresses")

selected_features_old <- c("type", "out_sum", "in_sum", "out_mean", "in_mean", "lifetime", "delay_mean", "delay_min", "delay_max", "act_day",
                            "max_tx_per_day","in_gini", "out_degree", "in_degree")

#selected_features <- c("type", "out_sum", "in_sum")


xm <- subset(dat, select = selected_features) # BEWARE !! can be dat, training or testing
xm <- subset(xm, select = -c(type)) 
xm_scaled <- as.data.frame(scale(xm, center =TRUE, scale =TRUE))
#xm <- as.data.frame(scale(xm, center =TRUE, scale =TRUE)) 
# sum(is.na(dat$in_gini)) 

# mahalanobis distance
#install.packages("robustbase")
library(robustbase)
library(rrcov)
# change number of outliers to find
kk = 50 # number of outliers to be found 
# xxxx <- mahalanobis(xm_scaled, colMeans(xm_scaled), CovMcd(xm_scaled)@cov)
xxxx <- mahalanobis(xm_scaled, colMeans(xm_scaled), cov(xm_scaled))
sorted_mahal <- sort(xxxx, decr=T)  # by values
order_mahal <- order(xxxx, decreasing =T)  # by lines 
cutoff = sort(xxxx, decr=T)[kk] # cutoff on the VALUE (not on the number of outliers)

x_final <- dat
x_final$outliers <- factor(ifelse(xxxx >= cutoff, "TRUE", "FALSE"))
x_final$maha_score <- xxxx
x_final_decreasing  <- x_final[order(-x_final$maha_score),] # descending order
prop.table(table(x_final$outliers))

# use "ICSOutlier" 
library("ICSOutlier")
xm_ics <- ics2(xm_scaled)
out_ics <- ics.distances(xm_ics)
# set a threshold, for example outICS > = 100 to define as outlier 
#x_final <- x_final %>% mutate(ID = rownames(x_final), outICS = out_ics, outliers_ICS = (outICS >= 100) ) # be careful with ID should be the same row
x_final_ics <- x_final %>% mutate( outICS = out_ics, outliers_ICS = (outICS >= 100) ) 

# so this ICSoutliers is simply the Mahalnobis distance ! ---> not useful

# check with kmeans
dat_cluster <- kmeans(as.matrix(xm_scaled), 4, nstart =2)
print(dat_cluster$centers)
dev.new()
plot(out_degree ~in_degree, data =xm_scaled, log ="xy", col=as.factor(dat_cluster$cluster))
RSD2 <- sd(dat_cluster$cluster)/mean(dat_cluster$cluster)
print(RSD2)
table(dat_cluster$cluster, dat$type) # horrible! ---> not useful

# pam partitioning around medoids
library(cluster)
library(ineq)
cluster_pam <- pam(xm,4)
dev.new()
plot(out_degree ~ in_degree, data = xm,  log ="xy", col= as.factor(cluster_pam$clustering))
points(cluster_pam$medoids, pch=19)
table(cluster_pam$clustering, dat$type)

# Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
# fast implementation of Hierarchical DBSCAN (HDBSCAN)
#install.packages("dbscan")
#  minPts= minimum number of data points needed in a neighborhood to define a cluster
library("dbscan")
cdb <- hdbscan(xm_scaled, minPts=50)
print(cdb)
plot(out_degree ~in_degree, data =xm_scaled, log ="xy", col=cdb$cluster+1, pch=20)
table(cdb$cluster, dat$type)
print(cdb$cluster_scores)
hullplot(xm_scaled, cdb)

x <- as.matrix(xm_scaled)

dev.new()
plot(in_gini, in_sum, dat= x)
glosh <- glosh(x, k=5)
summary(glosh)
dev.new()
hist(glosh, breaks = 10)
plot_glosh <- function(x, glosh){
  plot(x, pch = ".", main = "GLOSH (k=5)")
  points(x, cex = glosh*3, pch = 1, col="red")
  text(x[glosh > 0.80,], labels = round(glosh, 3)[glosh > 0.80], pos = 3)
}
plot_glosh(x, glosh)

x_final2 <- x_final %>% mutate( glosh = glosh )
x_final2  <- x_final2[order(-x_final2$maha_score),] # descending order
head(x_final2)
x_final3  <- x_final2[order(-x_final2$glosh),] # descending order
head(x_final3,20)

# use LOF Local Outlier Factor Score
# k size of the neighborhood
#my_lof <- lof(xm_scaled, k =5)
my_lof <- lof(as.matrix(xm), k =5)
summary(my_lof)
hist(my_lof, breaks=10)
dev.new()
plot(in_sum ~ in_gini, data =xm, log ="y")
points(in_sum ~ in_gini, data =xm, cex = (my_lof-1)*3, pch = 1, col="red")


# plot type
dev.new()
gg <- ggplot(x_final, aes(x=(in_gini), y=(in_sum), colour=type)) + geom_point() + scale_y_continuous(trans='log10', limits =c(0.1, 5e5)) + 
      scale_colour_manual(values = c("Blue", "Red", "Green", "Gray")) 
plot(gg)

# plot outliers
dev.new()
gg <- ggplot(x_final, aes(x=(in_gini), y=(in_sum), colour=outliers)) + geom_point() + scale_y_continuous(trans='log10', limits =c(0.1, 1e5)) + 
  scale_colour_manual(values = c("gray","red"))
plot(gg)
prop.table(table(x_final$type))


library(ineq)
# 
x_final_short <- subset(x_final, select = -c(type, address, in_sd, out_sd, outliers_ICS, outICS))

x_mean    <- x_final_short %>% group_by(outliers) %>% summarise_all(mean) # same as centers
x_entropy <- x_final_short %>% group_by(outliers) %>% summarise_all(entropy) 
x_ineq    <- x_final_short %>% group_by(outliers) %>% summarise_all(ineq, type="entropy") 
x_Gini    <- x_final_short %>% group_by(outliers) %>% summarise_all(Gini) 

#
x_final_short <- subset(x_final, select = -c(outliers, address, in_sd, out_sd, outliers, outliers_ICS, outICS))
type_mean    <- x_final_short %>% group_by(type) %>% summarise_all(mean) # same as centers
type_entropy <- x_final_short %>% group_by(type) %>%summarise_all(entropy) 
type_ineq    <- x_final_short %>% group_by(type) %>%summarise_all(ineq, type="entropy") 
type_Gini    <- x_final_short %>% group_by(type) %>%summarise_all(Gini) 

# write.csv(s_mean, file = "./dat_mean.csv")

# install.packages("huxtable")
library("huxtable")
hux(x_mean, add_colnames = TRUE)

# xtable
library("xtable")
xtable(x_mean, add_colnames = TRUE, caption = "average values for each anomaly....change digits")
xtable(type_mean, add_colnames = TRUE, caption = "average values for each anomaly....change digits")

print(xtable(type_mean, add_colnames = TRUE, caption = "average values for each anomaly....change digits"))
