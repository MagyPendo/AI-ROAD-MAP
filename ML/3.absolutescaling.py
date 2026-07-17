import pandas as pd 
import numpy as np
import matplotlib.pyplot  as plt

df = pd.read_csv('Titanic-Dataset.csv')


print((df / np.max(np.abs(df), axis=0)).head())