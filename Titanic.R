library(caret)
library(mlr)
library(e1071)
library(party)
library('randomForest')
library('caretEnsemble')

train <- read.csv('train.csv', header = TRUE, sep = ",")
test <- read.csv('test.csv',header = TRUE, sep = ",")
target <- train[,c('Survived')] 
train <- train [,-which(names(train) %in% c('Survived'))]
data <- rbind(train,test)


data_modified <- data[,c(2,4:7,9,11)]

data_modified$Pclass <- as.factor(data_modified$Pclass)
data_modified$SibSp <- as.factor(data_modified$SibSp)
data_modified$Parch <- as.factor(data_modified$Parch)
   
data_modified <- dummy.data.frame(data_modified, 
                                  names = c('Pclass', 'Embarked','SibSp','Parch'), sep = '')
preProcValues <- preProcess(data_modified, method = c("medianImpute","range"))
data_processed <- predict(preProcValues,data_modified)

train_modified <- data_processed[1:nrow(train),]
test_modified <- data_processed[(nrow(train)+1:nrow(test)),]
train_modified$Survived <- as.factor(target)

rows <- sample(nrow(train_modified),floor(nrow(train_modified)*0.70))
train_modified_train <- train_modified[rows,]
train_modified_test <- train_modified[-rows,]

logistic_regression <- function(train_modified_train, train_modified_test, test_modified){
  model <- glm(Survived ~ Pclass1 + Pclass2 + Pclass3 + Sex + Age + Fare + Parch0 +
                 SibSp0 + SibSp1 + EmbarkedS, train_modified_train,family = "binomial")
  Survived_Predicted_Train <- predict(model,train_modified_train,type = "response")
  Survived_Predicted_Train_Values  <- ifelse(Survived_Predicted_Train > 0.5,1,0)
  Survived_Predicted_Test <- predict(model,train_modified_test,type = "response")
  Survived_Predicted_Test_Values  <- ifelse(Survived_Predicted_Test > 0.5,1,0)
  confusion_matrix_train <- confusionMatrix(Survived_Predicted_Train_Values,train_modified_train$Survived)
  confusion_matrix_test <- confusionMatrix(Survived_Predicted_Test_Values,train_modified_test$Survived)
  accuracy_train_logistic <- confusion_matrix_train$overall['Accuracy']
  accuracy_test_logistic <- confusion_matrix_test$overall['Accuracy']
  Survived_Test_Values <- predict(model,test_modified,type = "response")
  Survived_test<- ifelse(Survived_Test_Values > 0.5,1,0)
  print(cat("Accuracy on Training dataset for Logistic Regression is :- ", accuracy_train_logistic))
  print(cat("Accuracy on Training dataset for Logistic Regression is :- ", accuracy_test_logistic))

  test$Survived <- Survived_test
  submission <- test[,c(1,12)]
  write.csv(submission,file='test_predictions_logistic.csv',row.names = FALSE)
  
}

svm <- function(train_modified_train,train_modified_test,test_modified){
  #model_svm <- tune.svm(Survived ~ Pclass_X1 + Pclass_X2 + Pclass_X3 + Sex + Age + Fare + Parch_X1 +
  #                        SibSp_X1 + SibSp_X2 + Embarked_X1,
  #                       data = train_modified_train, kernel= 'radial',gamma = 10^(-5:-1), cost = 10^(-3:-1))
  model_svm <- svm(Survived ~ Pclass1 + Pclass2 + Pclass3 + Sex + Age + Fare + Parch0 +
                     SibSp0 + SibSp1 + EmbarkedS, data = train_modified_train, kernel= "radial")
  Survived_SVM_Train <- predict(model_svm,train_modified_train)
  Survived_SVM_Test <- predict(model_svm,train_modified_test)
  confusion_matrix_svm_train <- confusionMatrix(Survived_SVM_Train,train_modified_train$Survived)
  confusion_matrix_svm_test <- confusionMatrix(Survived_SVM_Test,train_modified_test$Survived)
  accuracy_train_svm <- confusion_matrix_svm_train$overall['Accuracy']
  accuracy_test_svm <- confusion_matrix_svm_test$overall['Accuracy']
  Survived_SVM <- predict(model_svm,test_modified)
  print(cat("Accuracy on Training dataset for SVM is :- ", accuracy_train_svm))
  print(cat("Accuracy on Training dataset for SVM is :- ", accuracy_test_svm))

  test$Survived <- Survived_SVM
  submission <- test[,c(1,12)]
  write.csv(submission,file='test_predictions_SVM.csv',row.names = FALSE)
  
}

ctrees <- function (train_modified_train, train_modified_test, test_modified){
  model_trees <- ctree(Survived ~ Pclass1 + Pclass2 + Pclass3 + Sex + Age + Fare + Parch0 +
                         SibSp0 + SibSp1 + EmbarkedS, train_modified_train)
  Survived_trees_train <- predict(model_trees,train_modified_train)
  Survived_trees_test <- predict(model_trees,train_modified_test)
  confusion_matrix_trees_train <- confusionMatrix(Survived_trees_train,train_modified_train$Survived)
  confusion_matrix_trees_test <- confusionMatrix(Survived_trees_test,train_modified_test$Survived)
  accuracy_train_trees <- confusion_matrix_trees_train$overall['Accuracy']
  accuracy_test_trees <- confusion_matrix_trees_test$overall['Accuracy']
  Survived_Trees <- predict(model_trees,test_modified)
  
  print(cat("Accuracy on Training dataset for ctree is :- ", accuracy_train_trees))
  print(cat("Accuracy on Validation dataset for ctree is :- ", accuracy_test_trees))  
  
  test$Survived <- Survived_Trees
  submission <- test[,c(1,12)]
  write.csv(submission,file='test_predictions_Trees.csv',row.names = FALSE)
  
}
  
ensemble <- function(train_modified){

  feature.names=names(train_modified)[ncol(train_modified)]
  
  for (f in feature.names) {
    if (class(train_modified[[f]])=="factor") {
      levels <- unique(c(train_modified[[f]]))
      train_modified[[f]] <- factor(train_modified[[f]],
                                    labels=make.names(levels))
    }
  }
  
  control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
  models_list <- c('rpart', 'svmRadial' , 'glm')
  models <- caretList(Survived ~ Pclass1 + Pclass2 + Pclass3 + Sex + Age + Fare + Parch0 +
                        SibSp0 + SibSp1 + EmbarkedS, data = train_modified,
                      methodList=models_list,trControl = control)
  results <- resamples(models)
  summary(results)
  dotplot(results)
  
  stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
  stack.glm <- caretStack(models, method="glm", metric="Accuracy",trControl = stackControl)
  print(stack.glm)
  
  stack.rf <- caretStack(models, method="rf", metric="Accuracy",trControl = stackControl)
  print(stack.rf)  
  
}


logistic_regression(train_modified_train,train_modified_test,test_modified)
#svm(train_modified_train,train_modified_test,test_modified)
ctrees(train_modified_train,train_modified_test,test_modified)
