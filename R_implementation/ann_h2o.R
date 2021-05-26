library(caret)
library(h2o)

h2o.init(nthreads=-1, max_mem_size="2G")
## Jei netycia dar runina kazkoks klasteris
h2o.removeAll()

## Nuskaitome pradinius duomenis
train = read.csv("preprocessed_train.csv", stringsAsFactors=TRUE)
test = read.csv("preprocessed_test.csv", stringsAsFactors=TRUE)

train = train[,2:18]
test = test[,2:18]

train$y = as.factor(train$y)
test$y = as.factor(test$y)

## Paverciame juos i h2o objektus
h2o_train=as.h2o(train)
h2o_test=as.h2o(test)

## Apmokome modeli
dl_model1 = h2o.deeplearning(x = names(h2o_train[,1:16]),
                            y = "y",
                            training_frame = h2o_train,
                            activation="Rectifier",
                            hidden = c(10),
                            loss = "CrossEntropy",
                            score_each_iteration=TRUE,
                            epochs = 10000,
                            balance_classes=F,
                            rate=0.01,
                            adaptive_rate = F,
                            stopping_rounds=0,
                            classification_stop=-1)

plot(dl_model1)

#Predict and confusion table
prediction1 = as.data.frame(h2o.predict(object = dl_model1,
                                       newdata = h2o_test))
confMatrix1=table(true_labels=test$y, predicted_labels=prediction1[,1])
confMatrix1

#Accuracies for each class
diag(confMatrix1)/rowSums(confMatrix1)*100


## Isbandome kitoki modeli
dl_model2 = h2o.deeplearning(x = names(h2o_train[,1:16]),
                             y = "y",
                             training_frame = h2o_train,
                             activation="Tanh",
                             hidden = c(128,128,128),
                             loss = "CrossEntropy",
                             score_each_iteration=TRUE,
                             epochs = 20,
                             balance_classes=F,
                             adaptive_rate = F,
                             rate = 0.05,
                             stopping_rounds=0,
                             classification_stop=-1)

plot(dl_model2)

#Predict and confusion table
prediction2 = as.data.frame(h2o.predict(object = dl_model2,
                                        newdata = h2o_test))
confMatrix2=table(true_labels=test$y, predicted_labels=prediction2[,1])
confMatrix2

#Accuracies for each class
diag(confMatrix2)/rowSums(confMatrix2)*100
