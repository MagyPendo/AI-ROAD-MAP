import pandas as pd 
import numpy as np

df = pd.read_csv('Titanic-Dataset.csv')

print('Print dataframe Information')
df.info()
print('print dataframe head')
print(df.head())
print('printing duplicates')
print(df.duplicated())