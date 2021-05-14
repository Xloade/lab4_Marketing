# %% imports
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
# %% get dataset
df = pd.read_csv("../phpkIxskf.csv", sep=",")

# %% incoding categorical vars
ord_enc = OrdinalEncoder()
categorical = ["job","marital_status", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
df[categorical] = ord_enc.fit_transform(df[categorical])
# %% oversample
X = df.loc[ : , df.columns != 'y']
y = df['y']
categorical_index = map(df.columns.get_loc, categorical)
categorical_index = list(categorical_index)
X_resampled, y_resampled = SMOTENC(categorical_features=categorical_index, random_state=0).fit_resample(X, y)
df_resampled = pd.concat([X_resampled,y_resampled], axis=1)
# %% oversampling plots
sns.lmplot(x='balance', y='duration', data=df, hue='y', fit_reg=False)
plt.title("Before oversampling")
plt.show()
sns.lmplot(x='balance', y='duration', data=df_resampled, hue='y', fit_reg=False)
plt.title("After oversampling")
# %% saving changes
df.to_csv('preprocessed.csv')

# %%
