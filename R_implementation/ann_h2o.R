library(caret)
library(h2o)

h2o.init(nthreads=-1, max_mem_size="2G")
## Jei netycia dar runina kazkoks klasteris
h2o.removeAll()

data=read.csv("preprocessed.csv", stringsAsFactors=TRUE)
str(data)

## Pradinius duomenis padaliname i apmokymo ir testavimo
idx.train = createDataPartition(y = data$y, p = 0.8, list =FALSE)
train = data[idx.train, ]
test = data[-idx.train, ]

train$y = as.factor(train$y)
test$y = as.factor(test$y)

## Paverciame juos i h2o objektus
h2o_train=as.h2o(train)
h2o_test=as.h2o(test)

## Apmokome modeli
dl_model = h2o.deeplearning(x = names(h2o_train[,1:16]),
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

plot(dl_model)

#Predict and confusion table
prediction = as.data.frame(h2o.predict(object = dl_model,
                                       newdata = h2o_test))
confMatrix=table(true_labels=test$y, predicted_labels=prediction[,1])
confMatrix

#Accuracies for each class
diag(confMatrix)/rowSums(confMatrix)*100
