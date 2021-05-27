# %% import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
# %% read data
train = pd.read_csv("../preprocessed_train.csv", usecols=range(1,18))
test = pd.read_csv("../preprocessed_test.csv", usecols=range(1,18))
# %%
train.head()
# %%
X_train = train[train.columns.difference(['y'])]
y_train = train['y']
X_test = test[test.columns.difference(['y'])]
y_test = test['y']
# %%
print(len(y_train[y_train == 2]))
print(len(y_train[y_train == 1]))
print(len(y_test[y_test == 2]))
print(len(y_test[y_test == 1]))
# %%
clf = LogisticRegression(
    solver='saga',
    class_weight='balanced',
    random_state=0
    ).fit(X_train, y_train)
# %%
y_pred = clf.predict(X_test)
confmat = confusion_matrix(y_test, y_pred)
print(confmat)
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred, average=None)
print("Bendras tikslumas:", acc)
print("Klasių tikslumai:", rec)
# %%
threshold = 0.4
decisions = (clf.predict_proba(X_test)[:,1] >= threshold).astype(int) + 1
confmat = confusion_matrix(y_test, decisions)
print(confmat)
acc = accuracy_score(y_test, decisions)
rec = recall_score(y_test, decisions, average=None)
print("Bendras tikslumas:", acc)
print("Klasių tikslumai:", rec)
# %%
