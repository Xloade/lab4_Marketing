# %% imports
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
# %% get dataset
df = pd.read_csv("../phpkIxskf.csv", sep=",")
categorical = [col for col in df.columns if df[col].dtype=="O"]
continuous = [col for col in df.columns if df[col].dtype=="int64"]
continuous.remove("y")
# %% incoding categorical vars
ord_enc = OrdinalEncoder()
df[categorical] = ord_enc.fit_transform(df[categorical])
# %% splitting data
X = df.loc[ : , df.columns != 'y']
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#%% plot before preprocessing
sns.lmplot(x='balance', y='duration', data=pd.concat([X_train,y_train], axis=1), hue='y', fit_reg=False, scatter_kws={ "alpha":0.2})
plt.title("Before preprocessing")
plt.show()
print("class balance before preprocessing:")
y_train.value_counts()
# %% removing outliers
outliers = (np.abs(stats.zscore(X_train[continuous])) < 3).all(axis=1)
X_train = X_train[outliers]
y_train = y_train[outliers]
# %%
scaler = StandardScaler()
scaler.fit(X_train)
X_train[X_train.columns] = scaler.transform(X_train)
X_test[X_test.columns] = scaler.transform(X_test)
# %% oversample
categorical_index = map(df.columns.get_loc, categorical)
categorical_index = list(categorical_index)
X_resampled, y_resampled = SMOTENC(categorical_features=categorical_index, random_state=0).fit_resample(X_train, y_train)
df_resampled = pd.concat([X_resampled,y_resampled], axis=1)

# %% plot after preprocessing
sns.lmplot(x='balance', y='duration', data=df_resampled, hue='y', fit_reg=False, scatter_kws={ "alpha":0.2})
plt.title("After preprocessing")
plt.show()
print("class balance after preprocessing:")
y_resampled.value_counts()
# %% saving changes
df_resampled.to_csv('preprocessed_train.csv')
pd.concat([X_test,y_test], axis=1).to_csv('preprocessed_test.csv')
# %%
