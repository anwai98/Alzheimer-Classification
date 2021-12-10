# Calling Libraries (and installed the rest)
library(caret)
library(MASS)
library(e1071)
library(usdm)
library(nortest)
library(rlang)
library(devtools)
library(glmnet)
library(ggpubr)
# install_github("vqv/ggbiplot")
# library(ggbiplot)

set.seed(400)

###########################################################################################################
################################### ADCN CLASSIFICATION ###################################################
###########################################################################################################

# Importing the Training and Testing data
ADCN_training_data <- read.csv(file = 'ADCNtrain.csv')
ADCN_testing_data <- read.csv(file = 'ADCNtest.csv')

# Binarize the Labels
ADCN_training_labels <- ifelse(ADCN_training_data$Labels == "AD",1,0)
ADCN_training_labels <- as.factor(ADCN_training_labels)

# Removing the 'Patient ID' and 'Labels' column from Training and Testing data (the index column)
ADCN_training_data <- ADCN_training_data[-1]
ADCN_training_data <- ADCN_training_data[-567]
ADCN_testing_ID <- ADCN_testing_data[1]
ADCN_testing_data <- ADCN_testing_data[-1]

# # Normality Checking
# set.seed(1234)
# dplyr::sample_n(ADCN_training_data, 10)
# ggdensity(ADCN_training_data$G_Insula.anterior.1.L, main = "Density plot of Training Data", xlab = "Training Data")
# ggqqplot(ADCN_training_data$G_Insula.anterior.1.L)
# shapiro.test(ADCN_training_data$G_Insula.anterior.1.L)

# # Collinearity
# colinearity <- vifcor(ADCN_training_data)

# # 33 variables from the 566 input variables have collinearity problem:
# SLC6A8 SELENBP1 SLC6A10P....SLC6A8....SLC6A10PB.
# TRIM58 TSPAN5 FECH SLC25A39 PIM1 GMPR S_Anterior_Rostral.1.R KLF1 RNF10 PIP4K2A STRADB DMTN
# BCL2L1 N_Caudate.3.R G_subcallosal.1.R TNS1 PITHD1 N_Putamen.2.R TMOD1 SNCA
# G_Paracentral_Lobule.3.R DCAF12 GLRX5 G_Fusiform.3.R S_Sup_Frontal.1.R YBX3 IFIT1B
# G_Hippocampus.2.R S_Sup_Frontal.2.L N_Thalamus.9.R

# Removing Unwanted Collinear Features from the training dataset
ADCN_training_data <- within(ADCN_training_data, rm("SLC6A8", "SELENBP1", "SLC6A10P....SLC6A8....SLC6A10PB.","TRIM58","TSPAN5","FECH","SLC25A39","PIM1", "GMPR", "S_Anterior_Rostral.1.R", "KLF1", "RNF10", "PIP4K2A", "STRADB", "DMTN", "BCL2L1", "N_Caudate.3.R", "G_subcallosal.1.R", "TNS1", "PITHD1", "N_Putamen.2.R", "TMOD1", "SNCA", "G_Paracentral_Lobule.3.R", "DCAF12", "GLRX5", "G_Fusiform.3.R", "S_Sup_Frontal.1.R","YBX3", "IFIT1B", "G_Hippocampus.2.R", "S_Sup_Frontal.2.L", "N_Thalamus.9.R" ))

# Correlation of Features
feature_correlation <- cor(ADCN_training_data)
index <- findCorrelation(feature_correlation, .8) # the index of the columns to be removed because they have a high correlation
removed_features <- colnames(feature_correlation)[index] # the name of the columns chosen above
ADCN_training_data <- ADCN_training_data[!names(ADCN_training_data) %in% removed_features] # now go back to df and use removed_features to subset the original data frame

######## LASSO Regression for Feature Selection ############
x <- as.matrix(ADCN_training_data) # all X vars
# Fit the LASSO model (Lasso: Alpha = 1)
set.seed(100)
cv.lasso <- cv.glmnet(x, ADCN_training_labels, family='binomial', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc')
# Results
# plot(cv.lasso)
# plot(cv.lasso$glmnet.fit, xvar="lambda", label=TRUE)
cat('Min Lambda: ', cv.lasso$lambda.min, '\n 1Sd Lambda: ', cv.lasso$lambda.1se)
df_coef <- round(as.matrix(coef(cv.lasso, s=cv.lasso$lambda.min)), 2)
# See all contributing variables
df_coef <- df_coef[df_coef[, 1] == 0,]
# # Labels Back in the Dataset
ADCN_training_data$Labels <- ADCN_training_labels
ADCN_training_data <- ADCN_training_data[!names(ADCN_training_data) %in% names(df_coef)]

# # PCA
# ADCN_PCA <- prcomp(ADCN_training_data, scale.= TRUE)
# ADCN_PCA$rotation
# ggbiplot(ADCN_PCA, choices=c(1,2), groups=c("AD", "CN"), ellipse=True)
# ggbiplot(ADCN_PCA, choices=c(1,3), groups=c("AD", "CN"), ellipse=True)
# ggbiplot(ADCN_PCA, choices=c(1,4), groups=c("AD", "CN"), ellipse=True)
# ggbiplot(ADCN_PCA, choices=c(2,3), groups=c("AD", "CN"), ellipse=True)

# Subject ID of the Patients
ADCN.best_model <- subset(ADCN_testing_ID)

# # Cross Validation
trainControl <- trainControl(method="repeatedcv", number = 5, repeats = 10, sampling = "rose")
# SVM
# ADCN_SVM.fit <- train(Labels ~ ., data = ADCN_training_data, method="svmLinear", trControl = trainControl, preProcess = c("center", "scale"))
# ADCN_SVM.predict <- predict(ADCN_SVM.fit, ADCN_training_data)
# # KNN
# ADCN_KNN.fit <- train(Labels ~ ., data = ADCN_training_data, method="knn", trControl = trainControl, preProcess = c("center", "scale", "pca"))
# ADCN_KNN.predict <- predict(ADCN_KNN.fit, ADCN_training_data)
# LR - Testing
ADCN_LR.fit <- train(Labels ~ ., data = ADCN_training_data, method="glm", trControl = trainControl, preProcess = c("center", "scale", "pca"))
# ADCN_LR.predict <- predict(ADCN_LR.fit, ADCN_training_data)
ADCN_LR.predict <- predict(ADCN_LR.fit, ADCN_testing_data)
# # LDA
# ADCN_LDA.fit <- train(Labels ~ ., data = ADCN_training_data, method="lda", trControl = trainControl, preProcess = c("center", "scale"))
# ADCN_LDA.predict <- predict(ADCN_LDA.fit, ADCN_training_data)

# Evaluation Metrics
# mccr(ADCN_training_labels, ADCN_SVM.predict)
# auc(ADCN_training_labels, ADCN_SVM.predict)
# mccr(ADCN_training_labels, ADCN_KNN.predict)
# auc(ADCN_training_labels, ADCN_KNN.predict)
# mccr(ADCN_training_labels, ADCN_LDA.predict)
# auc(ADCN_training_labels, ADCN_LDA.predict)
# mccr(ADCN_training_labels, ADCN_LR.predict)
# auc(ADCN_training_labels, ADCN_LR.predict)

# # Analysing the Results
# summary(ADCN_training_data$Labels)
# summary(ADCN_SVM.predict)
# summary(ADCN_KNN.predict)
# summary(ADCN_LR.predict)
# summary(ADCN_LDA.predict)

# Labelling Back the Disease Stage
ADCN_LR.predict <- ifelse(ADCN_LR.predict == "1","AD", "CN")
ADCN.best_model$Labels <- ADCN_LR.predict

# Names of the Best Features
ADCN.best_features <- ADCN_training_data[, !names(ADCN_training_data) %in% "Labels"]
ADCN.best_features <- names(ADCN.best_features)

# Saving the Model
save(ADCN.best_model, file = "0063771_ARCHIT_challenge2_ADCNres.Rdata")
save(ADCN.best_features, file = "0063771_ARCHIT_challenge2_ADCNfeat.Rdata")

###########################################################################################################
################################### ADMCI CLASSIFICATION ##################################################
###########################################################################################################

# Importing the Training and Testing data
ADMCI_training_data <- read.csv(file = 'ADMCItrain.csv')
ADMCI_testing_data <- read.csv(file = 'ADMCItest.csv')

# Binarize the Labels
ADMCI_training_labels <- ifelse(ADMCI_training_data$Labels == "AD",1,0)
ADMCI_training_labels <- as.factor(ADMCI_training_labels)

# Removing the 'Patient ID' and 'Labels' column from Training and Testing data (the index column)
ADMCI_training_data <- ADMCI_training_data[-1]
ADMCI_training_data <- ADMCI_training_data[-597]
ADMCI_testing_ID <- ADMCI_testing_data[1]
ADMCI_testing_data <- ADMCI_testing_data[-1]

# # Normality Checking
# set.seed(1234)
# dplyr::sample_n(ADMCI_training_data, 10)
# ggdensity(ADMCI_training_data$G_Frontal_Sup.2.R, main = "Density plot of Training Data", xlab = "Training Data")
# ggqqplot(ADMCI_training_data$G_Frontal_Sup.2.R)
# shapiro.test(ADMCI_training_data$G_Frontal_Sup.2.R)

# # Collinearity
# colinearity <- vifcor(ADMCI_training_data)

# 40 variables from the 596 input variables have collinearity problem:
# SLC6A8 UBXN6 GMPR STRADB TSPAN5 SLC6A10P....SLC6A8....SLC6A10PB. TMOD1 SNCA N_Thalamus.9.R RNF10
# SELENBP1 FECH PIM1 G_Paracentral_Lobule.3.R G_Frontal_Med_Orb.1.R SLC25A39 TRIM58 G_subcallosal.1.R GUK1
# GLRX5 DMTN DCAF12 TNS1 G_Frontal_Sup_Orb.1.R N_Putamen.2.R N_Caudate.3.R PIP4K2A S_Anterior_Rostral.1.R
# S_Orbital.1.R S_Olfactory.1.L G_Frontal_Sup_Orb.1.L S_Sup_Frontal.2.L PITHD1 TESC YBX3 G_Fusiform.1.R
# G_Fusiform.3.R BCL2L1 G_ParaHippocampal.1.L DPM2

# Removing Unwanted Collinear Features from the training dataset
ADMCI_training_data <- within(ADMCI_training_data, rm("SLC6A8", "UBXN6", "GMPR", "STRADB", "TSPAN5", "SLC6A10P....SLC6A8....SLC6A10PB.", "TMOD1", "SNCA", "N_Thalamus.9.R", "RNF10", "SELENBP1", "FECH", "PIM1", "G_Paracentral_Lobule.3.R", "G_Frontal_Med_Orb.1.R", "SLC25A39", "TRIM58", "G_subcallosal.1.R", "GUK1", "GLRX5", "DMTN", "DCAF12", "TNS1", "G_Frontal_Sup_Orb.1.R", "N_Putamen.2.R", "N_Caudate.3.R", "PIP4K2A", "S_Anterior_Rostral.1.R", "S_Orbital.1.R", "S_Olfactory.1.L", "G_Frontal_Sup_Orb.1.L", "S_Sup_Frontal.2.L", "PITHD1", "TESC", "YBX3", "G_Fusiform.1.R", "G_Fusiform.3.R", "BCL2L1", "G_ParaHippocampal.1.L", "DPM2"))

# Correlation of Features
feature_correlation <- cor(ADMCI_training_data)
index <- findCorrelation(feature_correlation, .8) # the index of the columns to be removed because they have a high correlation
removed_features <- colnames(feature_correlation)[index] #the name of the columns chosen above
ADMCI_training_data <- ADMCI_training_data[!names(ADMCI_training_data) %in% removed_features] #now go back to df and use removed_features to subset the original data frame

######## LASSO Regression for Feature Selection ############
x <- as.matrix(ADMCI_training_data) # all X vars
# Fit the LASSO model (Lasso: Alpha = 1)
set.seed(100)
cv.lasso <- cv.glmnet(x, ADMCI_training_labels, family='binomial', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc')
# Results
# plot(cv.lasso)
# plot(cv.lasso$glmnet.fit, xvar="lambda", label=TRUE)
cat('Min Lambda: ', cv.lasso$lambda.min, '\n 1Sd Lambda: ', cv.lasso$lambda.1se)
df_coef <- round(as.matrix(coef(cv.lasso, s=cv.lasso$lambda.min)), 2)
# See all contributing variables
df_coef <- df_coef[df_coef[, 1] == 0,]
# # Labels Back in the Dataset
ADMCI_training_data$Labels <- ADMCI_training_labels
ADMCI_training_data <- ADMCI_training_data[!names(ADMCI_training_data) %in% names(df_coef)]

# # PCA
# ADMCI_PCA <- prcomp(ADMCI_training_data, scale.= TRUE)
# ADMCI_PCA$rotation
# ggbiplot(ADMCI_PCA, choices=c(1,2), groups=c("AD", "MCI"), ellipse=True)
# ggbiplot(ADMCI_PCA, choices=c(1,3), groups=c("AD", "MCI"), ellipse=True)
# ggbiplot(ADMCI_PCA, choices=c(1,4), groups=c("AD", "MCI"), ellipse=True)
# ggbiplot(ADMCI_PCA, choices=c(2,3), groups=c("AD", "MCI"), ellipse=True)

# Subject ID of the Patients
ADMCI.best_model <- subset(ADMCI_testing_ID)

# # Cross Validation
trainControl <- trainControl(method="repeatedcv", number = 5, repeats = 10, sampling = "rose")
# # SVM
# ADMCI_SVM.fit <- train(Labels ~ ., data = ADMCI_training_data, method="svmLinear", trControl = trainControl, preProcess = c("center", "scale", "pca"))
# ADMCI_SVM.predict <- predict(ADMCI_SVM.fit, ADMCI_training_data)
# # KNN
# ADMCI_KNN.fit <- train(Labels ~ ., data = ADMCI_training_data, method="knn", trControl = trainControl, preProcess = c("center", "scale", "pca"))
# ADMCI_KNN.predict <- predict(ADMCI_KNN.fit, ADMCI_training_data)
# LR
ADMCI_LR.fit <- train(Labels ~ ., data = ADMCI_training_data, method="glm", trControl = trainControl, preProcess = c("center", "scale"))
# ADMCI_LR.predict <- predict(ADMCI_LR.fit, ADMCI_training_data)
ADMCI_LR.predict <- predict(ADMCI_LR.fit, ADMCI_testing_data)
# # LDA
# ADMCI_LDA.fit <- train(Labels ~ ., data = ADMCI_training_data, method="lda", trControl = trainControl, preProcess = c("center", "scale", "pca"))
# ADMCI_LDA.predict <- predict(ADMCI_LDA.fit, ADMCI_training_data)

# Evaluation Metrics
# mccr(ADMCI_training_labels, ADMCI_SVM.predict)
# auc(ADMCI_training_labels, ADMCI_SVM.predict)
# mccr(ADMCI_training_labels, ADMCI_KNN.predict)
# auc(ADMCI_training_labels, ADMCI_KNN.predict)
# mccr(ADMCI_training_labels, ADMCI_LDA.predict)
# auc(ADMCI_training_labels, ADMCI_LDA.predict)
# mccr(ADMCI_training_labels, ADMCI_LR.predict)
# auc(ADMCI_training_labels, ADMCI_LR.predict)

# Analysing the Results
# summary(ADMCI_training_data$Labels)
# summary(ADMCI_SVM.predict)
# summary(ADMCI_KNN.predict)
# summary(ADMCI_LR.predict)
# summary(ADMCI_LDA.predict)

# Labelling Back the Disease Stage
ADMCI_LR.predict <- ifelse(ADMCI_LR.predict == "1","AD", "MCI")
ADMCI.best_model$Labels <- ADMCI_LR.predict

# Names of the Best Features
ADMCI.best_features <- ADMCI_training_data[, !names(ADMCI_training_data) %in% "Labels"]
ADMCI.best_features <- names(ADMCI.best_features)

# Saving the Model
save(ADMCI.best_model, file = "0063771_ARCHIT_challenge2_ADMCIres.Rdata")
save(ADMCI.best_features, file = "0063771_ARCHIT_challenge2_ADMCIfeat.Rdata")

###########################################################################################################
################################### MCICN CLASSIFICATION ##################################################
###########################################################################################################

# Importing the Training and Testing data
MCICN_training_data <- read.csv(file = 'MCICNtrain.csv')
MCICN_testing_data <- read.csv(file = 'MCICNtest.csv')

# Binarize the Labels
MCICN_training_labels <- ifelse(MCICN_training_data$Labels == "MCI",1,0)
MCICN_training_labels <- as.factor(MCICN_training_labels)

# Removing the 'Patient ID' and 'Labels' column from Training and Testing data (the index column)
MCICN_training_data <- MCICN_training_data[-1]
MCICN_training_data <- MCICN_training_data[-422]
MCICN_testing_ID <- MCICN_testing_data[1]
MCICN_testing_data <- MCICN_testing_data[-1]

# # Normality Checking
# set.seed(1234)
# dplyr::sample_n(MCICN_training_data, 10)
# ggdensity(MCICN_training_data$Angular_L, main = "Density plot of Training Data", xlab = "Training Data")
# ggqqplot(MCICN_training_data$Angular_L)
# shapiro.test(MCICN_training_data$Angular_L)

# # Collinearity
# colinearity <- vifcor(MCICN_training_data)
# 58 variables from the 421 input variables have collinearity problem:
# Amygdala_R IGHG2 ENSG00000211896....ENSG00000211897....ENSG00000233855 ENSG00000211896 ENSG00000211890....ENSG00000211895 IGHV4.31 ENSG00000211893....ENSG00000211896....ENSG00000211897....ENSG00
# ENSG00000211890 KDM5D XIST IGHG3 EIF1AY ORM1 ENSG00000211895 KIR2DS2....LOC100996743....KIR2DL3....KIR2DS1....KIR2DS4 DDX3Y APOBEC3B SLC6A8 IGHA1 TMEM176B HBG1....HBG2
# LAIR2 PRKY ENSG00000231486....ENSG00000239975....ENSG00000242076 EPB42 LCN2 KLRC1....KLRC2 ENSG00000211625....ENSG00000239951 RSAD2 IFIT1 Cerebelum_9_L
# IFI44 Frontal_Med_Orb_R HBD GMPR Cerebelum_8_R FAM46C Cerebelum_7b_L Rectus_L RPS4Y1 BPGM Cerebelum_6_L Frontal_Sup_2_R SLC4A1 GSTM1....GSTM2....GSTM4....GSTM5....GSTM2P1
# ISG15 ANK1 HERC5 SELENBP1 Hippocampus_R OAS3 Putamen_L Lingual_L Insula_R TMOD1 Precuneus_R NEK7 HLA.DQB1

# Removing Unwanted Collinear Features from the training dataset
MCICN_training_data <- within(MCICN_training_data, rm("Amygdala_R", "IGHG2", "ENSG00000211896....ENSG00000211897....ENSG00000233855", "ENSG00000211896", "ENSG00000211890....ENSG00000211895", "IGHV4.31", "ENSG00000211893....ENSG00000211896....ENSG00000211897....ENSG00", "ENSG00000211890", "KDM5D", "XIST", "IGHG3", "EIF1AY", "ORM1", "ENSG00000211895", "KIR2DS2....LOC100996743....KIR2DL3....KIR2DS1....KIR2DS4", "DDX3Y", "APOBEC3B", "SLC6A8", "IGHA1", "TMEM176B", "HBG1....HBG2", "LAIR2", "PRKY", "ENSG00000231486....ENSG00000239975....ENSG00000242076", "EPB42", "LCN2", "KLRC1....KLRC2", "ENSG00000211625....ENSG00000239951", "RSAD2", "IFIT1", "Cerebelum_9_L", "IFI44", "Frontal_Med_Orb_R", "HBD", "GMPR", "Cerebelum_8_R", "FAM46C", "Cerebelum_7b_L", "Rectus_L", "RPS4Y1", "BPGM", "Cerebelum_6_L", "Frontal_Sup_2_R", "SLC4A1", "GSTM1....GSTM2....GSTM4....GSTM5....GSTM2P1", "ISG15", "ANK1", "HERC5", "SELENBP1", "Hippocampus_R", "OAS3", "Putamen_L", "Lingual_L", "Insula_R", "TMOD1", "Precuneus_R", "NEK7", "HLA.DQB1"))

# Correlation of Features
feature_correlation <- cor(MCICN_training_data)
index <- findCorrelation(feature_correlation, .9) # the index of the columns to be removed because they have a high correlation
removed_features <- colnames(feature_correlation)[index] #the name of the columns chosen above
MCICN_training_data <- MCICN_training_data[!names(MCICN_training_data) %in% removed_features] #now go back to df and use removed_features to subset the original data frame

######## LASSO Regression for Feature Selection ############
x <- as.matrix(MCICN_training_data) # all X vars
# Fit the LASSO model (Lasso: Alpha = 1)
set.seed(100)
cv.lasso <- cv.glmnet(x, MCICN_training_labels, family='binomial', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc')
# Results
# plot(cv.lasso)
# plot(cv.lasso$glmnet.fit, xvar="lambda", label=TRUE)
cat('Min Lambda: ', cv.lasso$lambda.min, '\n 1Sd Lambda: ', cv.lasso$lambda.1se)
df_coef <- round(as.matrix(coef(cv.lasso, s=cv.lasso$lambda.min)), 2)
# See all contributing variables
df_coef <- df_coef[df_coef[, 1] == 0,]
# # Labels Back in the Dataset
MCICN_training_data$Labels <- MCICN_training_labels
MCICN_training_data <- MCICN_training_data[!names(MCICN_training_data) %in% names(df_coef)]

# # PCA
# MCICN_PCA <- prcomp(MCICN_training_data, scale.= TRUE)
# MCICN_PCA$rotation
# ggbiplot(MCICN_PCA, choices=c(1,2), groups=c("MCI", "CN"), ellipse=True)
# ggbiplot(MCICN_PCA, choices=c(1,3), groups=c("MCI", "CN"), ellipse=True)
# ggbiplot(MCICN_PCA, choices=c(1,4), groups=c("MCI", "CN"), ellipse=True)
# ggbiplot(MCICN_PCA, choices=c(2,3), groups=c("MCI", "CN"), ellipse=True)

# Subject ID of the Patients
MCICN.best_model <- subset(MCICN_testing_ID)

# # Cross Validation
trainControl <- trainControl(method="repeatedcv", number = 5, repeats = 10)
# SVM
MCICN_SVM.fit <- train(Labels ~ ., data = MCICN_training_data, method="svmLinear", trControl = trainControl, preProcess = c("center", "scale"))
# MCICN_SVM.predict <- predict(MCICN_SVM.fit, MCICN_training_data)
MCICN_SVM.predict <- predict(MCICN_SVM.fit, MCICN_testing_data)
# # KNN
# MCICN_KNN.fit <- train(Labels ~ ., data = MCICN_training_data, method="knn", trControl = trainControl, preProcess = c("center", "scale"))
# MCICN_KNN.predict <- predict(MCICN_KNN.fit, MCICN_training_data)
# # LR
# MCICN_LR.fit <- train(Labels ~ ., data = MCICN_training_data, method="glm", trControl = trainControl, preProcess = c("center", "scale"))
# MCICN_LR.predict <- predict(MCICN_LR.fit, MCICN_training_data)
# # LDA
# MCICN_LDA.fit <- train(Labels ~ ., data = MCICN_training_data, method="lda", trControl = trainControl, preProcess = c("center", "scale"))
# MCICN_LDA.predict <- predict(MCICN_LDA.fit, MCICN_training_data)

# Evaluation Metrics
# mccr(MCICN_training_labels, MCICN_SVM.predict)
# auc(MCICN_training_labels, MCICN_SVM.predict)
# mccr(MCICN_training_labels, MCICN_KNN.predict)
# auc(MCICN_training_labels, MCICN_KNN.predict)
# mccr(MCICN_training_labels, MCICN_LDA.predict)
# auc(MCICN_training_labels, MCICN_LDA.predict)
# mccr(MCICN_training_labels, MCICN_LR.predict)
# auc(MCICN_training_labels, MCICN_LR.predict)

# Analysing the Results
# summary(MCICN_training_data$Labels)
# summary(MCICN_SVM.predict)
# summary(MCICN_KNN.predict)
# summary(MCICN_LR.predict)
# summary(MCICN_LDA.predict)

# Labelling Back the Disease Stage
MCICN_SVM.predict <- ifelse(MCICN_SVM.predict == "1","MCI", "CN")
MCICN.best_model$Labels <- MCICN_SVM.predict

# Names of the Best Features
MCICN.best_features <- MCICN_training_data[, !names(MCICN_training_data) %in% "Labels"]
MCICN.best_features <- names(MCICN.best_features)

# Saving the Model
save(MCICN.best_model, file = "0063771_ARCHIT_challenge2_MCICNres.Rdata")
save(MCICN.best_features, file = "0063771_ARCHIT_challenge2_MCICNfeat.Rdata")

