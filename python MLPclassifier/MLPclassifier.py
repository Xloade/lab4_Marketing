# %% get dataset
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

train = pd.read_csv("../preprocessed_train.csv", sep=",")
test = pd.read_csv("../preprocessed_test.csv", sep=",")
# %% set data
X_train = train.loc[ : , train.columns != 'y']
X_test = test.loc[ : , test.columns != 'y']
Y_train = train["y"]
Y_test = test["y"]
# %% train and fit
classifier = MLPClassifier(hidden_layer_sizes=(150,100,50),learning_rate_init=0.0001, max_iter=1000,activation = 'logistic',solver='adam',random_state=1)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_pred, Y_test)
print(cm)
# %%
plt.plot(classifier.loss_curve_)
# %%
