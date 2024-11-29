import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import argparse

"""
ensemble method with Adaboost and Random forest
"""

# Command-line argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train an ensemble model with AdaBoost and RandomForest")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--output_model', type=str, default='ensemble_model.pkl', help="Path to save the trained model")
    return parser.parse_args()

# Load dataset
def load_data(data_path):
    return pd.read_csv(data_path)

# Drop columns with high missing values
def drop_high_missing_columns(df, threshold=50):
    missing_ratio = (df.isnull().sum() / len(df)) * 100
    return df.loc[:, missing_ratio < threshold]

def main():
    # Parse arguments
    args = parse_args()
    
    # Load data
    df = load_data(args.data_path)

    df = drop_high_missing_columns(df)

    columns_to_drop = [
        'homepage', 'tagline', 'backdrop_path', 'production_companies',
        'production_countries', 'spoken_languages', 'poster_path', 'overview'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Ensure the target column 'vote_average' exists and handle missing values
    if 'vote_average' in df.columns:
        df = df.dropna(subset=['vote_average'])
        y = df['vote_average']
        X = df.drop(columns=['vote_average'])
    else:
        raise ValueError("The target column 'vote_average' is not present in the dataset.")

    # Impute missing values for numerical and categorical columns
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = X.select_dtypes(include=['object']).columns

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', num_imputer),
                ('scaler', StandardScaler())
            ]), numerical_columns),
            ('cat', Pipeline(steps=[
                ('imputer', cat_imputer),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_columns)
        ]
    )

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create ensemble model with AdaBoost and RandomForest
    adaboost = AdaBoostRegressor(n_estimators=100, random_state=42)
    random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

    # Voting Regressor to combine both models
    ensemble_model = VotingRegressor(estimators=[
        ('adaboost', adaboost),
        ('random_forest', random_forest)
    ])

    # Create a complete pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', ensemble_model)
    ])

    # Set up parameter grid for GridSearchCV
    param_grid = {
        'model__adaboost__n_estimators': [50, 100, 150],
        'model__random_forest__n_estimators': [50, 100, 150],
        'model__random_forest__max_depth': [None, 10, 20]
    }

    # Perform Grid Search
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search
    best_pipeline = grid_search.best_estimator_

    # Save the trained model
    joblib.dump(best_pipeline, args.output_model)

    # Predict and evaluate the model
    y_pred = best_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    main()
