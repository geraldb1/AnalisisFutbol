import pandas as pd
df = pd.read_csv('data/processed/data_modeling.csv')
print(df['temporada'].dtype)
print(df['temporada'].unique()[:5])