import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df=pd.read_csv("Advertising.csv")
print(df.head())
df.drop("Unnamed: 0",axis=1,inplace=True)
print(df.info())
print(df.isnull().sum())
plt.scatter(df["TV"],df["Sales"])
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.title("TV vs Sales")
plt.show()
plt.scatter(df["Radio"],df["Sales"])
plt.xlabel("Radio Advertising")
plt.ylabel("Sales")
plt.title("Radio vs Sales")
plt.show()
plt.scatter(df["Newspaper"],df["Sales"])
plt.xlabel("Newspaper Advertising")
plt.ylabel("Sales")
plt.title("Newspaper vs Sales")
plt.show()
x=df[["TV","Radio","Newspaper"]]
y=df["Sales"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,
                                               random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
score=r2_score(y_test,y_pred)
print("R2 Score:", score)
tv = float(input("Enter TV Advertising Budget: "))
radio = float(input("Enter Radio Advertising Budget: "))
newspaper = float(input("Enter Newspaper Advertising Budget: "))
new_data = [[tv, radio, newspaper]]
prediction = model.predict(new_data)
print("Predicted Sales:", prediction[0])