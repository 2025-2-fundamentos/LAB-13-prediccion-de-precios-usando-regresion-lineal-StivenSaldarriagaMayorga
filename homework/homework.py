import gzip
import pickle
from pathlib import Path
import os
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

train_pd = pd.read_csv("files/input/train_data.csv.zip", compression="zip").copy()
test_pd = pd.read_csv("files/input/test_data.csv.zip", compression="zip").copy()
train_pd["Age"] = 2021 - train_pd["Year"]
test_pd["Age"] = 2021 - test_pd["Year"]
train_pd = train_pd.drop(columns=["Year", "Car_Name"])
test_pd = test_pd.drop(columns=["Year", "Car_Name"])
X_train = train_pd.drop(columns=["Present_Price"])
y_train = train_pd["Present_Price"]
X_test = test_pd.drop(columns=["Present_Price"])
y_test = test_pd["Present_Price"]

cat_cols = ["Fuel_Type", "Selling_type", "Transmission"]
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocesador = ColumnTransformer(transformers=[("cat", OneHotEncoder(), cat_cols),("num", MinMaxScaler(), num_cols),])

pipe = Pipeline(steps=[("pre", preprocesador),("selector", SelectKBest(score_func=f_regression)),("reg", LinearRegression()),])

param_grid = {
    "selector__k": range(1, 15),
    "reg__fit_intercept": [True, False],
    "reg__positive": [True, False],
}

grid = GridSearchCV(estimator=pipe,param_grid=param_grid,cv=10,scoring="neg_mean_absolute_error",n_jobs=-1,refit=True,)
grid.fit(X_train, y_train)

os.makedirs("files/models", exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid, f)

pred_train = grid.predict(X_train)
pred_test = grid.predict(X_test)
train_metrics = {
    "type": "metrics",
    "dataset": "train",
    "r2": float(r2_score(y_train, pred_train)),
    "mse": float(mean_squared_error(y_train, pred_train)),
    "mad": float(median_absolute_error(y_train, pred_train)),
}
test_metrics = {
    "type": "metrics",
    "dataset": "test",
    "r2": float(r2_score(y_test, pred_test)),
    "mse": float(mean_squared_error(y_test, pred_test)),
    "mad": float(median_absolute_error(y_test, pred_test)),
}

Path("files/output").mkdir(parents=True, exist_ok=True)

with open("files/output/metrics.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(train_metrics) + "\n")
    f.write(json.dumps(test_metrics) + "\n")