import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv("../phpkIxskf.csv", sep=",")
profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
profile.to_file("your_report.html")
