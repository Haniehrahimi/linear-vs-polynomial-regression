#Imports
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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
