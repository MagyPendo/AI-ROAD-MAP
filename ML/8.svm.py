import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC


df = pd.read_csv('student_training_data.csv')

print(df.info())
print(df.head())

# 2. Encode Categorical Data (SVM cannot process text strings)
df_encoded = pd.get_dummies(
    df, columns=["fee_payment_status", "reporting_status"], drop_first=True
)
# X=df[['fee_payment_status','exam_performance','reporting_status']]
# y=df['needs_assistance']

# 3. DecisionBoundaryDisplay requires exactly 2 numerical features
# Let's pick 'exam_performance' and one of our new dummy encoded features
features = ["exam_performance", "fee_payment_status_Unpaid"]
X = df_encoded[features]
y = df_encoded["needs_assistance"]

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=0)

svm=SVC(kernel="linear", C=1)
svm.fit(X_train,y_train)


DecisionBoundaryDisplay.from_estimator(
        svm,
        X_train,
        response_method="predict",
        alpha=0.8,
        cmap="Pastel1",
        xlabel=features[0],
        ylabel=features[1],
    )

plt.scatter(
            X_train.iloc[:, 0],
            X_train.iloc[:, 1],
            c=y_train,   
            s=20, edgecolors="k")
plt.show()
