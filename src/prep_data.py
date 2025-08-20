import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def prepare_data():
    # Load dataset
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['median_house_value'] = housing.target * 100000  # Scale to dollars for realism

    # Handle missing values (none in this dataset, but for completeness)
    df.fillna(df.median(), inplace=True)

    # No need for rooms_per_household since AveRooms is already rooms per household
    # If TotalRooms is needed, it could be approximated as df['AveRooms'] * df['Households']
    # But we'll skip redundant feature engineering for simplicity

    # Scale features
    features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Split and save (for training)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    df.to_csv('data/cal_housing.csv', index=False)  # Raw processed data

    print("Data prepared and saved.")

if __name__ == "__main__":
    prepare_data()

# DVC optional: dvc add data/cal_housing.csv; dvc push