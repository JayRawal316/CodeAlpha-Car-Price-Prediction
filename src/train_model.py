import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import RandomizedSearchCV
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_csv('data/car data.csv')
print(df.columns)

#creating car age
current_year = datetime.now().year

df['Car_Age'] = current_year - df['Year']

#drop years
df.drop(['Year'], axis=1, inplace=True)

#convert to numbers for machine understanding -
df = pd.get_dummies(df,columns=['Fuel_Type','Selling_type','Transmission'], drop_first=True)

#drop Car_Name as it is not useful for prediction
df.drop(['Car_Name'], axis=1, inplace=True)

#Train and Test Split - 
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']
#Split -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training Multiple Models - 

#Linear Regression -
lr = LinearRegression()
lr.fit(X_train, y_train)

#Descision Tree Regressor -
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

#Random Forest Regressor -
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

#Gradient Boosting Regressor -
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

#Model Evalutaion -

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    return mae, mse, rmse, r2
#Evaluate all models

print("Linear Regression")
print(evaluate_model(lr, X_test, y_test))

print("Decision Tree")
print(evaluate_model(dt, X_test, y_test))

print("Random Forest")
print(evaluate_model(rf, X_test, y_test))

print("Gradient Boosting")
print(evaluate_model(gb, X_test, y_test))

#Hyper-Parameter Tuning - 
params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5,10,15,None],
    "min_samples_split": [2, 5, 10],
}
search = RandomizedSearchCV(estimator=rf, param_distributions=params, n_iter=10, cv=5, n_jobs=-1, random_state=42)
search.fit(X_train, y_train)

#Feature Importance - 
importance = gb.feature_importances_
plt.figure(figsize=(10,5))
sns.barplot(
    x=importance,
    y=X.columns
)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("images/feature_importance.png")
plt.show()

#Plot - 
importance = gb.feature_importances_

plt.figure(figsize=(10,5))

sns.barplot(
    x=importance,
    y=X.columns
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("images/feature_importance.png")

plt.show()

#save best model - 
joblib.dump(
    gb,
    "models/car_price_model.pkl"
)
print("Model saved Successfully!")

# Actual vs Predicted Plot
y_pred = gb.predict(X_test)
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
    linewidth=2
)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price")
plt.tight_layout()
plt.savefig("images/actual_vs_predicted.png", dpi=300)
print("Actual vs Predicted image saved.")
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Selling Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("images/residual_plot.png", dpi=300)
print("Residual Plot image saved.")
plt.show()