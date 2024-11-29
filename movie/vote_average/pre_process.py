import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

# Load the CSV file into a DataFrame
def load_data(file_path):
    return pd.read_csv(file_path)

# Data Cleaning and Preparation using ColumnTransformer
def preprocess_data(data):
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = data.select_dtypes(include=['object', 'bool']).columns

    # Reduce high-cardinality categorical features by grouping less frequent categories as 'Other'
    for column in categorical_columns:
        if data[column].nunique() > 50:  # Threshold for high-cardinality categorical features
            top_categories = data[column].value_counts().nlargest(50).index
            data[column] = data[column].apply(lambda x: x if x in top_categories else 'Other')

    # Data Cleaning and Preparation using ColumnTransformer
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', num_imputer),
                ('scaler', MinMaxScaler())
            ]), numerical_columns),
            ('cat', Pipeline(steps=[
                ('imputer', cat_imputer),
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]), categorical_columns)
        ]
    )

    # Fit and transform the data
    data_cleaned = preprocessor.fit_transform(data)

    # Convert cleaned data back to DataFrame format
    numerical_data = pd.DataFrame(data_cleaned[:, :len(numerical_columns)], columns=numerical_columns)
    if len(categorical_columns) > 0:
        categorical_data = pd.DataFrame(data_cleaned[:, len(numerical_columns):], columns=categorical_columns)
        data_cleaned = pd.concat([numerical_data, categorical_data], axis=1)
    else:
        data_cleaned = numerical_data

    return data_cleaned, numerical_columns, categorical_columns

# Feature Selection using SelectKBest and L1 Regularization (Lasso)
def feature_selection(data_cleaned, numerical_columns):
    # Assuming 'vote_average' is the target column
    target_column = 'vote_average'
    X = data_cleaned.drop(columns=[target_column])
    y = data_cleaned[target_column]

    # Update numerical columns after preprocessing
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

    # SelectKBest with f_regression
    k_best = SelectKBest(score_func=f_regression, k=min(5, len(numerical_columns)))
    k_best.fit(X, y)
    selected_features = numerical_columns[k_best.get_support(indices=True)]

    # L1 Regularization (Lasso)
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X, y)
    l1_selector = SelectFromModel(lasso, prefit=True)
    l1_selected_features = numerical_columns[l1_selector.get_support(indices=True)]

    # Feature Importance using RandomForest
    forest = RandomForestRegressor(n_estimators=100, random_state=42)
    forest.fit(X, y)
    feature_importances_rf = pd.Series(forest.feature_importances_, index=numerical_columns)
    important_features = feature_importances_rf.nlargest(min(5, len(numerical_columns))).index

    # Combine selected features from different methods
    selected_features_combined = set(selected_features).union(set(l1_selected_features)).union(set(important_features))
    X_selected = X[list(selected_features_combined)]

    return X_selected, y

# Save cleaned data to a CSV file
def save_data(data_cleaned, output_path):
    data_cleaned.to_csv(output_path, index=False)

# Example usage
if __name__ == "__main__":
    # Load data
    file_path = '/Users/doa_ai/Developer/datamining/movie/data_raw.csv'
    data = load_data(file_path)

    # Preprocess data
    data_cleaned, numerical_columns, categorical_columns = preprocess_data(data)

    # Save cleaned data to a CSV file
    output_path = 'cleaned_data.csv'
    save_data(data_cleaned, output_path)

    # Feature selection
    X_selected, y = feature_selection(data_cleaned, numerical_columns)
    
    # Now X_selected and y can be used for training AdaBoost or ensemble models like RandomForest + AdaBoost
