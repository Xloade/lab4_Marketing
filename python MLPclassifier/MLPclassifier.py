# %% get dataset
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import joblib

train = pd.read_csv("../preprocessed_train.csv", sep=",")
test = pd.read_csv("../preprocessed_test.csv", sep=",")
# %% set data
X_train = train.loc[ : , train.columns != 'y']
X_train = X_train.loc[ : , X_train.columns != 'Unnamed: 0']
X_test = test.loc[ : , test.columns != 'y']
X_test = X_test.loc[ : , X_test.columns != 'Unnamed: 0']
Y_train = train["y"]
Y_test = test["y"]
# %% train and fit
classifier = MLPClassifier(hidden_layer_sizes=(150,100,50),learning_rate_init=0.005, max_iter=100,activation = 'logistic',solver='adam',random_state=1)
classifier.out_activation_
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(Y_test,y_pred)
print(cm)
# %%
plt.plot(classifier.loss_curve_)
# %% dump for API
joblib.dump(classifier, 'MLPClassifier.pkl', compress=1)
# %%
