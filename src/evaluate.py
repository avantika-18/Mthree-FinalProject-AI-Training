import mlflow
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import pandas as pd

def evaluate_registered_model(model_uri):
    model = mlflow.pyfunc.load_model(model_uri)
    df_test = pd.read_csv('data/test.csv')
    X_test = df_test.drop('median_house_value', axis=1)
    y_test = df_test['median_house_value']
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)  # Updated to use root_mean_squared_error
    r2 = r2_score(y_test, predictions)
    
    print(f"Evaluation - MAE: {mae}, RMSE: {rmse}, R2: {r2}")

if __name__ == "__main__":
    # Example: Evaluate production model
    evaluate_registered_model("models:/California_Housing_Prediction@production")