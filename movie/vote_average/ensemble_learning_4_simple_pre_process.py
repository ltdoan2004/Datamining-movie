import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
"""
ensemble method with AdaBoost, RandomForest, GradientBoosting, and ExtraTrees.
"""
# Argument parser for data path
def parse_arguments():
    parser = argparse.ArgumentParser(description="Clean and train ensemble model on dataset")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the input CSV dataset")
    parser.add_argument('--output_model', type=str, default='ensemble_model.pkl', help="Path to save the trained model")
    return parser.parse_args()

# Drop columns with a high percentage of missing values
def drop_high_missing_columns(df, threshold=50):
    missing_ratio = (df.isnull().sum() / len(df)) * 100
    return df.loc[:, missing_ratio < threshold]

# Main function
def main(data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Drop columns with high missing values
    df = drop_high_missing_columns(df)

    # Drop irrelevant columns
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

    # Create ensemble models with AdaBoost, RandomForest, GradientBoosting, and ExtraTrees
    adaboost = AdaBoostRegressor(random_state=42)
    random_forest = RandomForestRegressor(random_state=42)
    gb_regressor = GradientBoostingRegressor(random_state=42)
    extra_trees = ExtraTreesRegressor(random_state=42)

    # Parameter grid for tuning
    param_grid = {
        'model__adaboost__n_estimators': [50, 100, 150],
        'model__random_forest__n_estimators': [100, 200, 300],
        'model__gradient_boosting__n_estimators': [100, 150, 200],
        'model__extra_trees__n_estimators': [100, 200, 300]
    }

    # Voting Regressor to combine multiple models
    ensemble_model = VotingRegressor(estimators=[
        ('adaboost', adaboost),
        ('random_forest', random_forest),
        ('gradient_boosting', gb_regressor),
        ('extra_trees', extra_trees)
    ])

    # Create a complete pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', ensemble_model)
    ])

    # Parameter tuning using GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Save the best model
    joblib.dump(grid_search.best_estimator_, 'args.output_model')

    # Predict and evaluate the model
    y_pred = grid_search.best_estimator_.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    args = parse_arguments()
    main(args.data_path)
