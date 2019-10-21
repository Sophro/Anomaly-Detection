# Anomaly-Detection
1.	Script1: read Bitcoin raw data provided by GraphSense and calculate useful features 
The script reads raw data such as transactions or timestamps as provided by GraphSense for each address/entity. From these basic data it produces useful features. 

2.	Script2: reduce the number of features 
This script reads all the features produced from the previous one. It checks for correlations between them and does a Principal Component Analysis. It tries different combination of features in order to select the most importants and reduce their number.

3.	Script3: compare different supervised methods to classify anomaly types
The script uses different classifying methods and selects the most efficient, optimizing the result also artifically reducing or increasing the sampling data.

4.	Script4: unsupervised methods to classify anomaly types
The script uses different unsupervised methods to classify anomaly types. Density-based as well as distance-based methods are used and compared.

5.	Script5: combine supervised and unsupervised methods to classify anomaly types
Selecting the most efficient supervised and unsuprvised methods anomaly tags are produced and compared with ground-truth. A final ranking score is provided and can be used to further investigate more suspicious activies.

