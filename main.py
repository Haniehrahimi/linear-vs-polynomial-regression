#Imports
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
#Load Dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

#Check Missing Values
print(df.isnull().sum())

#Correlation
print(df.corr()["MedHouseVal"].sort_values(ascending=False))

#Define X and y
X = df.drop("MedHouseVal",axis=1)
y = df["MedHouseVal"]

print(X,y)
# Train / Test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Linear Regression
model_linear = LinearRegression()

model_linear.fit(X_train,y_train)

y_pred_linear = model_linear.predict(X_test)

# Evaluate Linear Regression
mae_linear = mean_absolute_error(y_test,y_pred_linear)
mse_linear = mean_squared_error(y_test,y_pred_linear)
r2_linear = r2_score(y_test,y_pred_linear)

print("\n Linear Regression: ")
print("MAE: ",mae_linear)
print("MSE: ",mse_linear)
print("R2: ",r2_linear)

#Polynomial Regression
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model_poly = LinearRegression()
model_poly.fit(X_train_poly,y_train)

y_pred_poly = model_poly.predict(X_test_poly)

#Evaluate Polynomial Regression

mae_poly = mean_absolute_error(y_test,y_pred_poly)
mse_poly = mean_squared_error(y_test,y_pred_poly)
r2_poly = r2_score(y_test,y_pred_poly)

print("\nPolynomial Regression:")
print("MAE:", mae_poly)
print("MSE:", mse_poly)
print("R2:", r2_poly)

#Comparison Chart
models = ["Linear Regression","Polynomial Regression"]
r2_scores = [r2_linear,r2_poly]

plt.bar(models,r2_scores)

plt.ylabel("R2 Score")
plt.title("Linear vs Polynomial Regression")

plt.ylim(0,1)

plt.show()