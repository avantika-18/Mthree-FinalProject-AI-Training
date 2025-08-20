import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Local MLflow server
mlflow.set_experiment("California_Housing_Prediction")

def load_data():
    df = pd.read_csv('data/cal_housing.csv')
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(model, model_name, params):
    X_train, X_test, y_train, y_test = load_data()
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)  # Updated to use root_mean_squared_error
        r2 = r2_score(y_test, predictions)
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Log model
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        print(f"{model_name} - MAE: {mae}, RMSE: {rmse}, R2: {r2}")

if __name__ == "__main__":
    # Train models
    lr = LinearRegression()
    train_model(lr, "LinearRegression", {"fit_intercept": True})
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    train_model(rf, "RandomForest", {"n_estimators": 100, "max_depth": None})
    
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    train_model(xgb, "XGBoost", {"n_estimators": 100, "learning_rate": 0.1})