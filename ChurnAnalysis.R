
## Set up the directory
setwd("C:\\Users\\Varsha\\OneDrive\\pro\\churn")

## Load the data
data1 <- read.csv("C:\\Users\\Varsha\\OneDrive\\pro\\churn\\churn.csv")

## Install the required packages
library(caret)
library(rpart)
library(C50)
library(rattle)
library(party)
library(partykit)
library(randomForest)
library(ROCR)
library(ggplot2)
library(reshape2)
library(car)
library(corrplot)
library(e1071)

## Know the Data

dim(data1)
head(data1)
str(data1)
colnames(data1)
sum(is.na(data1))
class(data1)

      
###################
## Data Munging ###
###################
data1$Churn. <- as.integer(data1$Churn.)
data1$Int.l.Plan <- as.integer(data1$Int.l.Plan)
data1$VMail.Plan <- as.integer(data1$VMail.Plan)

data1$Churn.[data1$Churn.==1] <- 0
data1$Churn.[data1$Churn.==2] <- 1

data1$Int.l.Plan[data1$Int.l.Plan==1] <- 0
data1$Int.l.Plan[data1$Int.l.Plan==2] <- 1

data1$VMail.Plan[data1$VMail.Plan==1] <- 0
data1$VMail.Plan[data1$VMail.Plan==2] <- 1

#############################
## Drop unwanted variable ###
#############################
data1$State <- NULL
data1$Area.Code <- NULL
data1$Phone <- NULL

#############################
## Check for misssing data ##
#############################

summary(data1)
dim(data1)
na.omit(data1)
dim(data1)


###############################
## Exploratory Data Analysis ##
###############################

summary(data1)
sapply(data1, sd)

#correlation matrix
cormat <- round(cor(data1), digits = 2)
cormat

# heatmap of correlation using ggplot
qplot(x=Var1, y=Var2, data=melt(cor(data1, use="p")),
      fill=value, geom="tile") + scale_fill_gradient2(limits=c(-1,1))



## Histogram of day minutes
plot.new()
hist(data1$Day.Mins)

plot.new()
boxplot(data1$Day.Mins)
title("Boxplot of Day Min")



#######################################
## split dataset into train and test ##
#######################################

set.seed(1234)
## 70% training and 30% testing data
ind <- sample(2, nrow(data1), replace = TRUE, prob=c(0.7,0.3))
train <- data1[ind==1,]
test <- data1[ind==2,]

############################################################################################################

## Model 1 ##
## Logistic Regression ##

## select the variables to use based on forward selection procedure
## Lower AIC indicates better model

# forward Elimination

mod1 <- glm(Churn.~1, data = train)
biggest <- formula(glm(Churn.~., data = train))
biggest
forwardTest <- step(mod1, direction = "forward", scope = biggest)
modlogit <- glm(Churn. ~ Int.l.Plan + CustServ.Calls + Day.Charge + VMail.Plan + 
                  Eve.Mins + Intl.Charge + Intl.Calls + Night.Charge + VMail.Message + 
                  Night.Mins + Account.Length, family = "binomial", data = train)
summary(modlogit)

#influence Plot (clearly shows outliers)
influenceIndexPlot(modlogit, vars = c("cook", "hat"), id.n = 3)

##confidence interval
confint(modlogit)

# put the coefficients and confidence interval in a format onto a useful scale
exp(modlogit$coefficients)
exp(confint(modlogit))

## odds ratio only
exp(coef(modlogit))

## odds ratio and 95% CI
exp(cbind(OR=coef(modlogit), confint(modlogit)))


############################################################################################################

## Model 2 ##
## Support Vector Machine ##

svmModel <- svm(Churn.~., data=train, gamma=0.1, cost=1)
print(svmModel)
summary(svmModel)


############################################################################################################

## Model 3 ##
## Random Forest ##

randomForestModel <-randomForest(Churn.~., data = train, ntree=500, mtry=5, importance=TRUE)
print(randomForestModel)
importance(randomForestModel)

plot.new()
varImpPlot(randomForestModel, type = 1, pch = 19, col=1, cex=1.0, main = "")
abline(v=45, col="blue")

plot.new()
varImpPlot(randomForestModel, type = 2, pch=19, col=1, cex=1.0, main = "")
abline(v=45, col="blue")

############################################################################################################

## Model 4 ##
## Knowledge Discovery: Build a decision tree using C5.0 for churn ##

# the decision variable class must be converted into a factor
# the variable in order for C50 to process correctly

data1$Churn. <- as.factor(data1$Churn.)

#Run the C50 agorithm for decision tree

c50_tree <- C5.0(Churn.~., data = data1)

#display the summary
summary(c50_tree)
C5imp(c50_tree, metric = "usage")
C5imp(c50_tree, metric = "splits")

## run the C50 algorithm and show the decision rules
C50_rule_result <- C5.0(Churn.~., data = data1, rules=TRUE)
summary(C50_rule_result)


#######################################################################################

### Prediction

modlogitPred <- predict(modlogit, test, type = "response")
svmModelPred <- predict(svmModel, test, type = "response")
randomForestModelPred <- predict(randomForestModel, test, type = "response")

# this will create results as new column in a dataset
test$YHatLogit <- predict(modlogit, test, type = "response")
test$YHatSVM <- predict(svmModel, test, type = "response")
test$YHatRF <- predict(randomForestModel, test, type = "response")


## These are theshold parameter setting controls
predict1 <- function(t) ifelse(modlogitPred > t, 1, 0)
predict2 <- function(t) ifelse(svmModelPred > t, 1, 0)
predict3 <- function(t) ifelse(randomForestModelPred > t, 1, 0)

confusionMatrix(predict1(0.5), test$Churn.) ## Logistic Regression
table(predict1(0.5), test$Churn.)
##Accuracy
mean(predict1(0.5) == test$Churn.)## Accuracy of logit model 85%
confusionMatrix(predict2(0.5), test$Churn.) ## SVM Model
table(predict2(0.5), test$Churn.)
##Accuracy
mean(predict2(0.5) == test$Churn.)## Accuracy of SVM model 89%
confusionMatrix(predict3(0.5), test$Churn.) ## RandomForest
table(predict3(0.5), test$Churn.)
##Accuracy
mean(predict3(0.5) == test$Churn.)## Accuracy of RF model 95%


##########################
## ROC For Unpruned Model
##########################
LogitPrediction <- prediction(test$YHatLogit, test$Churn.)
SVMPrediction <- prediction(test$YHatSVM, test$Churn.)
RFPrediction <- prediction(test$YHatRF, test$Churn.)

perfLogit <- performance(LogitPrediction, "tpr", "fpr")
perfSVM <- performance(SVMPrediction, "tpr", "fpr")
perfRF <- performance(RFPrediction, "tpr", "fpr")

plot.new()
plot(perfLogit, col="green", lwd=2.5)
plot(perfSVM, add = TRUE, col ="blue", lwd=2.5)
plot(perfRF, add = TRUE, col = "orange", lwd=2.5)
abline(0,1,col="Red", lwd=2.5, lty=2)
title("ROC Curve")
legend(0.8,0.4,c("Logistic", "SVM", "Random Forest"), lty=c(1,1,1), 
       lwd = c(1.4,1.4,1.4), col=c("green", "blue", "orange"))
## We can see random forest is the appropriate model for this

### AUC(area under curve) calculation metrics

logit.auc <- performance(LogitPrediction, "auc")
svm.auc <- performance(SVMPrediction, "auc")
rf.auc <- performance(RFPrediction, "auc")

logit.auc #AUC=82.96%
svm.auc # AUC=92.58%
rf.auc # AUC=93.1%




