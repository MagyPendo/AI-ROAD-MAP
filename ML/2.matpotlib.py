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


# import matplotlib.pyplot as plt
# import numpy as np

xpoints = np.array([1, 8])
ypoints = np.array([3, 10])
# You can use the keyword argument marker to emphasize each point with a specified marker:
# You can use the keyword argument linestyle, or shorter ls, to change the style of the plotted line:
# Shorter Syntax
# The line style can be written in a shorter syntax:

# linestyle can be written as ls.

# dotted can be written as :.

# dashed can be written as --.
# You can use the keyword argument color or the shorter c to set the color of the line:

plt.plot(xpoints, ypoints, marker='*',linestyle='dotted',color='red')

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}
plt.title("Sports Watch Data", fontdict = font1)
plt.xlabel("Average Pulse", fontdict = font2)
plt.ylabel("Calorie Burnage", fontdict = font2)
plt.grid(axis = 'x')
plt.show()

# With Pyplot, you can use the grid() function to add grid lines to the plot.
# Set Line Properties for the Grid
# You can also set the line properties of the grid, like this: grid(color = 'color', linestyle = 'linestyle', linewidth = number).
y1=np.array([2,4,1,7,10])
y2=np.array([3,1,5,8])
plt.plot(y1,linestyle='dotted',color='red')
plt.plot(y2,linestyle='dashed',color='blue')
plt.title('Matplotlib')
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.show()

#plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(2, 1, 1)
plt.plot(x,y)
plt.title('Matplotlib')
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 1, 2)
plt.plot(x,y)
plt.title('Matplotlib')
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

plt.show()