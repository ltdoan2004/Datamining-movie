import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
from pre_process import *
import argparse
import time
from sklearn.utils import shuffle

"""
ensemble method with Adaboost and RandomForest
"""

# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an ensemble model with AdaBoost and RandomForest.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument('--output_model', type=str, default='ensemble_model.pkl', help="Path to save the trained model.")
    parser.add_argument('--tune_parameters', action='store_true', help="Whether to perform parameter tuning using GridSearchCV.")
    return parser.parse_args()

# Drop columns with high missing values
def drop_high_missing_columns(df, threshold=50):
    missing_ratio = (df.isnull().sum() / len(df)) * 100
    return df.loc[:, missing_ratio < threshold]

# Main function
def main():
    args = parse_arguments()

    # Load dataset
    df = pd.read_csv(args.data_path)
    
    # Drop columns with a high percentage of missing values
    df = drop_high_missing_columns(df)

    # Drop irrelevant columns
    columns_to_drop = [
        'homepage', 'tagline', 'backdrop_path', 'production_companies',
        'production_countries', 'spoken_languages', 'poster_path', 'overview'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Ensure the target column 'averageRating' exists and handle missing values
    if 'averageRating' in df.columns:
        df = df.dropna(subset=['averageRating'])
        y = df['averageRating']
        X = df.drop(columns=['averageRating'])
    else:
        raise ValueError("The target column 'averageRating' is not present in the dataset.")

    # Preprocess data
    data_cleaned, numerical_columns, categorical_columns = preprocess_data(df)

    # Feature selection
    X, y = feature_selection(data_cleaned, numerical_columns)

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
        ('model', ensemble_model)
    ])

    # Set up parameter grid for GridSearchCV
    param_grid = {
        'model__adaboost__n_estimators': [50, 100, 150],
        'model__random_forest__n_estimators': [50, 100, 150],
        'model__random_forest__max_depth': [None, 10, 20]
    }

    # Measure training time
    start_time = time.time()

    # Parameter tuning
    if args.tune_parameters:
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best Parameters: {best_params}")
    else:
        # Train the model without parameter tuning
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline
        best_params = None

    # Measure training completion time
    end_training_time = time.time()

    # Save the trained model
    joblib.dump(best_pipeline, args.output_model)

    # Predict and evaluate the model
    y_pred = best_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Set: {mse:.4f}")

    # Cross-validation evaluation
    start_cv_time = time.time()
    scores = cross_val_score(best_pipeline, X, y, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
    mse_scores = -scores
    end_cv_time = time.time()

    # Cross-validation metrics
    average_mse = np.mean(mse_scores)
    print(f"Mean Squared Error for each fold: {mse_scores}")
    print(f"Average Mean Squared Error (10-fold CV): {average_mse:.4f}")
    
    # Print run-time for training and evaluation
    training_time = end_training_time - start_time
    cv_time = end_cv_time - start_cv_time
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Cross-Validation Time: {cv_time:.2f} seconds")

    # Save results to a text file
    with open(f'{args.output_model}_results.txt', 'w') as result_file:
        result_file.write(f"Model Type: Ensemble (AdaBoost + RandomForest)\n")
        if best_params:
            result_file.write(f"Best Parameters: {best_params}\n")
        result_file.write(f"Mean Squared Error on Test Set: {mse:.4f}\n")
        result_file.write(f"Mean Squared Error for each fold: {mse_scores}\n")
        result_file.write(f"Average Mean Squared Error (10-fold CV): {average_mse:.4f}\n")
        result_file.write(f"Training Time: {training_time:.2f} seconds\n")
        result_file.write(f"Cross-Validation Time: {cv_time:.2f} seconds\n")

if __name__ == "__main__":
    main()
