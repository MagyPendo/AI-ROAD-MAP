import pandas as pd 
import numpy as np

df = pd.read_csv('Titanic-Dataset.csv')

print('Print dataframe Information')
df.info()
print('print dataframe head')
print(df.head())
print('printing duplicates')
print(df.duplicated())

#  Drop Irrelevant or Data-Heavy Missing Columns
# df.drop(columns=[]): Drops specified columns from the DataFrame.
# df.dropna(subset=[]): Removes rows where specified columns have missing values.
# fillna(): Fills missing values with specified value (e.g., mean).
print(df.drop(columns=['Name','Ticket','Cabin']))
print(df.dropna(subset=['Embarked'], inplace=True))
print(df['Age'].fillna(df['Age'].mean()))