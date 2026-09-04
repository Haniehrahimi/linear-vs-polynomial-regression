#Imports
import pandas as pd
from sklearn.datasets import fetch_california_housing


#Load Dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())