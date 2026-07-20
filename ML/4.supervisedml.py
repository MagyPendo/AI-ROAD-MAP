import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle as pk

# Loading a dataset
# df=pd.read_csv('')

data = load_iris()
X = data.data  
Y = data.target

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

dtree = DecisionTreeClassifier(max_depth=2, random_state=0)
dtree.fit(X_train, y_train)
dtree_preds = dtree.predict(X_test)
dtree_acc = accuracy_score(y_test, dtree_preds)
dtree_cm = confusion_matrix(y_test, dtree_preds)

print("Decision Tree Accuracy:", dtree_acc)
print("Decision Tree Confusion Matrix",dtree_cm)

plt.figure(figsize=(4, 3))
sns.heatmap(dtree_cm, annot=True, cmap="Blues", fmt="d")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

with open("model.pkl", "wb") as file:
    pk.dump(dtree, file)


