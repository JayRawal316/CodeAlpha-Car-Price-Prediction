import pandas as pd

df = pd.read_csv('data/car data.csv')
print(df.head())
print(df.info())
print(df.describe())