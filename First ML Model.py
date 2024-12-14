import numpy as np
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv")
print(f"df:\n{df}")

y = df["logS"]
print(f"y:\n{y}")
X = df.drop("logS", axis = 1)
print(f"Χ:\n{X}")



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
print(f"X_train:\n{X_train}")
print(f"X_test:\n{X_test}")

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
LinearRegression()
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)
print(y_lr_train_pred, y_lr_test_pred)
print(f"y_lr_train_pred:\n{y_lr_train_pred}")

print(f"y_lr_test_pred:\n{y_lr_test_pred}")

print(f"y_train:\n{y_train}")


print(f"y_lr_train_pred:\n{y_lr_train_pred}")

from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)
print(lr_train_mse)

print(lr_train_r2)

print(lr_test_mse)

print(lr_test_r2)

print('LR MSE (Train): ', lr_train_mse)

print('LR R2 (Train): ', lr_train_r2)

print('LR MSE (Test): ', lr_test_mse)

print('LR R2 (Test): ', lr_test_r2)

lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = [ 'Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2' ]
print(lr_results)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth = 2, random_state = 100)
rf.fit(X_train, y_train)
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = [ 'Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2' ]
print(rf_results)

df_models = pd.concat([rf_results, lr_results], axis=0)
print(df_models)

import matplotlib.pyplot as plt
plt.figure(figsize = (5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, color= 'green', alpha=0.3)

z= np.poly1d(np.polyfit(y_train, y_lr_train_pred, 1))

plt.plot(y_train, z(y_train), color='purple')
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')

plt.show()



