# AI-project in Linear Regression 
My first experience with machine learning involved developing a linear regression model using Python in PyCharm. The dataset I used was sourced online. The primary reason I chose to start with linear regression was to familiarize myself with fundamental concepts of machine learning, gain a deeper understanding of the model development process, and practically apply the knowledge I had acquired from the courses I was attending.

# Building the model
In my first machine learning project, I worked with a dataset related to the solubility of chemical compounds. The goal was to create a model that could predict the solubility of these compounds. Initially, I imported the dataset and defined my data frame. After that, I split the data into training and testing sets using an 80-20 ratio, with 80% for training and 20% for testing, in order to ensure proper evaluation of the model's performance. 

```python
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
```
Next, I trained the model and applied it to the test set to evaluate its performance. To assess this, I calculated the Mean Squared Error (MSE) and R-squared values for both the training and test sets, in order to determine how well the model was performing. As expected, the model was performing well.

```python
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
print('LR MSE (Train): ', lr_train_mse)

print('LR R2 (Train): ', lr_train_r2)

print('LR MSE (Test): ', lr_test_mse)

print('LR R2 (Test): ', lr_test_r2)

lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = [ 'Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2' ]
print(lr_results)
```
However, I decided to apply another ML Model, Random Forest, to compare which one would be more suitable for my data. After comparing, I realized that the linear regression model was the better choice.
```python
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
```
Finally, I visualized the relationship between the experimental LogS values (y_train) and the predicted values from the linear regression model (y_lr_train_pred) to better understand the model's performance.
```python
import matplotlib.pyplot as plt
plt.figure(figsize = (5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, color= 'green', alpha=0.3)

z= np.poly1d(np.polyfit(y_train, y_lr_train_pred, 1))

plt.plot(y_train, z(y_train), color='purple')
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')

plt.show()
```





