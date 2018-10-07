#Tried reading only a part of file as well using readBin(size estimation),purrr::map(selective reading)
#but equal distribution of all numbers was hard to get
#not adding that code to this file
#hence resolving to reading the whole file and then pickup sample from it
#setwd("C:/Users/Morien/Downloads/SVM_Dataset")
Train_total <- read.csv("mnist_train.csv", header = FALSE)
nrow(Train_total)#60000
num.col<-ncol(Train_total)
num.col#785

#summary(Train_total)
#str(Train_total)
#Data set contains 784 feature containg colour intersity decribing 1 feature
#generating columns names
col.names = paste0("V",seq(1,num.col))
col.names[1]='Number'
col.names
#assigning names to th colums
names(Train_total)<-col.names
Train_total$Number<-factor(Train_total$Number)
#backup to be used later
Train_total_bkp<-Train_total
#Train_total<-Train_total_bkp

#using only 15% of teh data selected randomly to train
set.seed(100)
library(caTools)
indices <- sample.split(Train_total$Number, SplitRatio = 0.85)
train <- Train_total[!(indices),]
summary(train[,"Number"])
# 0    1    2    3    4    5    6    7    8    9 
# 888 1011  894  920  876  813  888  940  878  892 
#we have good distribution of each number

############################################################
#loading test data
Final_Test <- read.csv("mnist_test.csv", header = FALSE)
nrow(Final_Test)#10000
ncol(Final_Test)#785
names(Final_Test)<-col.names
names(Final_Test)[1]
Final_Test$Number<-factor(Final_Test$Number)

#We will do the analysis on train and apply chnages to both the dataste
#i.e. Final_Test & train

#chekc for na
sum(is.na(train))
sum(is.na(Final_Test))
#sapply(train,function(x) sum(is.na(x)))
#sapply(Final_Test,function(x) sum(is.na(x)))
#data doesn't contain any na values

#Range of data in the features
max(train[,-1])#255
min(train[,-1])#0
#colour density ranging from 0-255

#checking unique value in each feature
unique_value<-sapply(train,function(x) length(unique(x)))
unique_value<-as.data.frame(cbind(value=as.numeric(unique_value),name=names(unique_value)))
str(unique_value)
unique_value[,"value"]<-as.numeric(unique_value[,"value"])
max(unique_value[,"value"])#178 unique vales
min(unique_value[,"value"])#single value represeting no data varaition

#single value features provide no variance hence dropping them
drops<-unique_value[which(unique_value$value==1 ),"name"]
length(drops)#111
train<-train[ , !(names(train) %in% drops)]
Final_Test<-Final_Test[ , !(names(Final_Test) %in% drops)]
#785-111=674
ncol(train)# 674

#lets try and remove features with low variablity further
var.data<-sapply(train[,-1],function(x) var(x))
#var.data[is.na(var.data)]
View(var.data)
plot(var.data)
plot(log(var.data))
#variance is very spread out from very low values to very high
#lets anlyze these further
var.plot<-quantile(var.data,probs = seq(0, 1, 0.01))
View(var.plot)
plot(var.plot,xaxt='n') 
axis(1, at = seq(5, 100, by = 5), las=2)
#we have a sigmoid curve with very low variace turing into very high between 30 to 50 percentile 
#we will removing all variable below 30 percentile , as they don't provide comparable prediction value 
#View(var.plot)
var.value<-var.plot[names(var.plot)=="30%"]
var.value#342.9711
# removing lower value records 
var.data<-var.data[var.data>var.value ]
View(var.data)
#Number the dependent varaible is not presnet in this list
#adding it manually
var.data$Number<-0
length(names(var.data))# 472 feature should remain
train<-train[ , (names(train) %in% names(var.data))]
Final_Test<-Final_Test[ , (names(Final_Test) %in% names(var.data))]
ncol(train)# 472


# We are not doing any ouliner treatment as the svm algorithm are immune to the problem
max(train[,-1])#255  
min(train[,-1])#0
#data is varing between not presenet to darkest value anyways which seems apt
#scaling we will do with the from withing the svm commnad

##############################################################
#from the recent wns hackthon realized that normally ditributed data doesn't offer much in term of predcition
#checking for these column being normal distribution
# Install required packages:
#install.packages('nortest')
library(nortest)
#Anderson-Darling normality test
#ad.test(combined_data$V14)$p.value

norm.data<-sapply(train[,-1],function(x) ad.test(x)$p.value)
View(norm.data)
max(norm.data)#3.7e-24
min(norm.data)#3.7e-24
# all p-value are pretty low , 
#implying the null hypothesis of data being nomraly ditributed can be rejected

#####################################################################################
#PCA
library(dummies)
 
pca.train <- train[indices,-1]
pca.test <- train[-indices,-1]
pca.train.scale <- scale(pca.train)
pca.train.scale<-na.omit(pca.train.scale)
sum(is.na(pca.train.scale))
lapply(pca.train, scale)
log(pca.train)
class(pca.train)
is.finite(as.data.frame(pca.train))
prin_comp <- prcomp(pca.train.scale, scale. = F)
names(prin_comp)
prin_comp

#outputs the mean of variables
prin_comp$center

#outputs the standard deviation of variables
prin_comp$scale
prin_comp$rotation
prin_comp$rotation[1:5,1:4]
dim(prin_comp$x)
biplot(prin_comp, scale = 0)

#compute standard deviation of each principal component
std_dev <- prin_comp$sdev

#compute variance
 pr_var <- std_dev^2

#check variance of first 10 components
 pr_var[1:10]
# [1] 40.54458 28.30936 25.94817 20.13106 18.18051 15.63609 12.76160 11.37960 10.35259  9.66205

 prop_varex <- pr_var/sum(pr_var)
 prop_varex[1:20]
 # [1] 0.08608192 0.06010480 0.05509165 0.04274110 0.03859982 0.03319764 0.02709470 0.02416051 0.02198003
 # [10] 0.02051391 0.01915622 0.01659524 0.01554749 0.01492929 0.01406910 0.01386465 0.01330045 0.01241830
 # [19] 0.01226078 0.01138652
 
 #scree plot
 plot(prop_varex, xlab = "Principal Component",
        ylab = "Proportion of Variance Explained",
        type = "b",xaxt='n')
axis(1, at = seq(5, 700, by = 20), las=2)
  
#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
       ylab = "Cumulative Proportion of Variance Explained",
       type = "b",xaxt='n')
axis(1, at = seq(5, 700, by = 20), las=2)
## we see that 30-40  components can describe teh data set retty well
## observation is similar to the varaice spread,
## we will try redcuing the actual number of varaibles instaed of using componenets

###################################################################################
#lets try to reduce the variabels space using factor analysis
#lets look at the correlation first
cor.data<-cor(train[,-1])
View(cor.data)
#Even via View command high co-relation values are evident
#corrplots take a lot of time
# library(corrplot)
# corrplot(cor.data)#not getting much info due to number of variable being high
max(cor.data[cor.data!=1])#0.8966463
min(cor.data)#-0.4356361


#############################################################
#Scale the data for factanal command
scaled_test <- as.data.frame(scale(train[,-1]))
#Factor Analysis
fac.test <- factanal(scaled_test, factors = 50,rotation = "varimax")
fac.test
# Test of the hypothesis that 50 factors are sufficient.
# The chi square statistic is 2580889 on 88360 degrees of freedom.
# The p-value is 0 

# Although 71% variace is covered by significance level is very low i.e. 0.
# representing co-planar nature of data;
# This indicates that the hypothesis of perfect model fit is rejected. 

#Lets try with higher number of factors
fac.test <- factanal(scaled_test, factors = 100,rotation = "varimax")
fac.test
# Test of the hypothesis that 100 factors are sufficient.
# The chi square statistic is 1282267 on 68535 degrees of freedom.
# The p-value is 0 
#again similar result confirming factor analysis is not the right approch for this data set

#####################################
ncol(train)#472

#Lets start with model building exercise with these set of variable
library(kernlab)
#Using Linear Kernel
Model_linear <- ksvm(Number~ ., data = train, scale = TRUE, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, Final_Test)
#confusion matrix - Linear Kernel
#install.packages("purrr")
library(purrr)
library(caret)
Linear.confMat<-confusionMatrix(Eval_linear,Final_Test$Number)
Linear.confMat
#Accuracy : 0.9166   
#91.66 accuracy is decent for this data

#####################
#Using RBF Kernel
#RBF kernal take steh logest time and is not select as teh final model 
#so feel free to ignore this section whiel evaluation
Model_RBF <- ksvm(Number~ ., data = train, scale = TRUE, kernel =rbfdot(sigma = 1))
Eval_RBF<- predict(Model_RBF, Final_Test)

RBF.confMat<-confusionMatrix(Eval_RBF,Final_Test$Number)
RBF.confMat
#Accuracy : 0.1135 
#accuracy is very low lets try increasing the sigma values

Model_RBF <- ksvm(Number~ ., data = train, scale = TRUE, kernel =rbfdot(sigma = 50))
Eval_RBF<- predict(Model_RBF, Final_Test)

RBF.confMat<-confusionMatrix(Eval_RBF,Final_Test$Number)
RBF.confMat
#Accuracy : 0.1135 
#No improvement signifying that thsi is not the right kernal to use

########################
#Using Polynomial kernal, degree 2 is being used to compare as 1 would make it linear
Model_POLY <- ksvm(Number~ ., data = train, scale = TRUE, kernel =polydot(degree = 2, scale = 1, offset = 1))
Eval_POLY<- predict(Model_POLY, Final_Test)

POLY.confMat<-confusionMatrix(Eval_POLY,Final_Test$Number)
POLY.confMat
#Accuracy : 0.96 
#Polynomial kernal has teh highest accuracy, we will use it for further processing

###################################################################
#plotting variance again
plot(var.plot,xaxt='n') 
axis(1, at = seq(5, 100, by = 5), las=2)
# we had eralier used 70% of the varaible for analysis
# lets check the performance the polynomial kernal on even lower set of feature

percentile.value<-c("40%","50%","60%","70%","80%","90%")
result.matrix<-c()
#i<-1
for (i in 1:length(percentile.value)){
  #data preparation
  var.value<-var.plot[names(var.plot)==percentile.value[i]]
  var.data.test<-var.data[var.data>var.value ]
  var.data.test$Number<-0
  train_test<-train[ , (names(train) %in% names(var.data.test))]
  #evaluation
  Model_POLY <- ksvm(Number~ ., data = train_test, scale = TRUE, kernel =polydot(degree = 2, scale = 1, offset = 1))
  Eval_POLY<- predict(Model_POLY, Final_Test)
  
  POLY.confMat<-confusionMatrix(Eval_POLY,Final_Test$Number)
  result.matrix<-rbind(result.matrix,c(percentile.value[i],POLY.confMat$overall["Accuracy"]))
}
colnames(result.matrix)[1]<-"Percentile_left_out"
View(result.matrix)
result.matrix
# #     Percentile_left_out Accuracy
# [1,] "40%"               "0.9621"
# [2,] "50%"               "0.9653"
# [3,] "60%"               "0.9656"
# [4,] "70%"               "0.9603"
# [5,] "80%"               "0.9424"
# [6,] "90%"               "0.8917"
#90% seem to have big drop in accuracy
#even with 80% of the feature removed accuracy is at a high 94%
#lets take the  complete data 20% high variance feature and see teh performce
# Train_total_bkp is where the complete backup  of the data was kept previsouly

var.value<-var.plot[names(var.plot)=="80%"]
var.data.test<-var.data[var.data>var.value ]
var.data.test$Number<-0
train_test<-Train_total_bkp[ , (names(Train_total_bkp) %in% names(var.data.test))]
ncol(train_test)#136
#evaluation
Model_POLY <- ksvm(Number~ ., data = train_test, scale = TRUE, kernel =polydot(degree = 2, scale = 1, offset = 1))
Eval_POLY<- predict(Model_POLY, Final_Test)

POLY.confMat<-confusionMatrix(Eval_POLY,Final_Test$Number)
POLY.confMat
# Accuracy : 0.9607  
# The model has an accuracy of 96% on test data

###################################################
# lets try and optymyte the polynomial kernal parameter using cross vaidation
# we will perform the test on 15% data only in view of time contraint

var.value<-var.plot[names(var.plot)=="80%"]
var.data.test<-var.data[var.data>var.value ]
var.data.test$Number<-0
train_test<-train[ , (names(train) %in% names(var.data.test))]

############   Hyperparameter tuning and Cross Validation #####################

# We will use the train function from caret package to perform Cross Validation. 

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 5 implies Number of folds in CV.
trainControl <- trainControl(method="cv", number=5)

# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.
metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

set.seed(100)
grid <- expand.grid(.degree=seq(2,5,1),.scale =T,.C=c(0.1,1,3,5) )
grid


#train function takes Number~ ., Data=train_test, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.
#this takes a lot of time 45 min- 1 hr
fit.svm <- train(Number~ ., data=train_test, method="svmPoly", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)
# Tuning parameter 'scale' was held constant at a value of TRUE
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were degree = 3, scale = TRUE and C = 0.1.
plot(fit.svm)


###################################
#final model has the follwing parameter
#degree = 3, scale = TRUE and C = 0.1
#we will be using only 20% of high variance features
#selecting 20 percitile varaible from total train data set
train_test<-Train_total_bkp[ , (names(Train_total_bkp) %in% names(var.data.test))]
ncol(train_test)#136
Model_POLY <- ksvm(Number~ ., data = train_test,C=0.1, scale = TRUE, kernel =polydot(degree = 3, scale = 1, offset = 1))
Eval_POLY<- predict(Model_POLY, Final_Test)

POLY.confMat<-confusionMatrix(Eval_POLY,Final_Test$Number)
POLY.confMat
#Accuracy : 0.9706
#Our final model has an accuracy of 97.06% on the test data while using only 20% of the variables

