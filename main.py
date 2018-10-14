import pandas
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

df = pandas.read_csv("./dataset_Facebook.csv",sep=";")
Y = df.get("Total Interactions")[:499]
Y_test = df.get("Total Interactions")[451:499]
df.pop("Post Month")
df.pop("Type")
df.pop("like")
df.pop("share")
df.pop("comment")
df.pop("Total Interactions")
X = df[:499]
X_test = df[451:499]
Xs=1

lrModel = LinearRegression()
scores = cross_val_score(lrModel, X, Y, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)


lrLasso = Lasso()
scores = cross_val_score(lrLasso, X, Y, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)