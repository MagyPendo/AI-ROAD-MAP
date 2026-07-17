import pandas as pd 
import numpy as np
import matplotlib.pyplot  as plt

df = pd.read_csv('Titanic-Dataset.csv')

plt.boxplot(df['Age'])
plt.ylabel('Vaiable')
plt.xlabel('Age')
plt.title('Box plot')
plt.show()

df.plot.box()
plt.ylabel('y-axis')
plt.xlabel('x-axis')
plt.show()