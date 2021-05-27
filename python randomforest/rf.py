# %% import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import RandomizedSearchCV
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
    n_estimators=200,
    max_features=2,
    max_depth=None,
    random_state=0
    )
clf.fit(X_train, y_train)
# %%
y_pred = clf.predict(X_test)
confmat = confusion_matrix(y_test, y_pred)
print(confmat)
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred, average=None)
print("Bendras tikslumas:", acc)
print("Klasių tikslumai:", rec)
# %%
threshold = 0.295
decisions = (clf.predict_proba(X_test)[:,1] >= threshold).astype(int) + 1
confmat = confusion_matrix(y_test, decisions)
print(confmat)
acc = accuracy_score(y_test, decisions)
rec = recall_score(y_test, decisions, average=None)
print("Bendras tikslumas:", acc)
print("Klasių tikslumai:", rec)
# %%
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['sqrt', 'log2', 2, 6, 8]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

random_grid = {#'n_estimators': n_estimators,
               'max_features': [int(x) for x in np.linspace(2, 11, num = 10)]
               #'max_depth': max_depth
               }

# %%
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

# %%
rf_random.cv_results_
# %%
'''
{'n_estimators': 400,
 'min_samples_split': 2,
 'min_samples_leaf': 1,
 'max_features': 'sqrt',
 'max_depth': None,
 'bootstrap': False}
 '''

 
rf_random.best_params_

# {'n_estimators': 200, 'max_features': 2, 'max_depth': None}

# %%
