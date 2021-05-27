# %% import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None, 
    random_state=0
    )
clf.fit(X_train, y_train)
# %%
y_pred = clf.predict(X_test)
confmat = confusion_matrix(y_test, y_pred)
print(confmat)
acc = accuracy_score(y_test, y_pred)
acc
# %%
