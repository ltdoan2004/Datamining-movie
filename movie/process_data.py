import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, mean_absolute_error
import joblib
import numpy as np
X_data = pd.read_csv('/Users/doa_ai/Developer/datamining/data_X.csv', sep=',')

Y_train = pd.read_csv('/Users/doa_ai/Developer/datamining/data_Y.csv', sep=',')
Y_submit = pd.read_csv('/Users/doa_ai/Developer/datamining/sample_submission.csv', sep=',')

train_df = X_data.merge(Y_train, left_on='date_time', right_on='date_time')
test_df = X_data.merge(Y_submit, left_on='date_time', right_on='date_time').drop('quality', axis=1)
y = train_df['quality']
train_df.drop(['quality'], axis=1, inplace=True)


train_df.drop(['date_time'], axis=1, inplace=True)
test_df.drop(['date_time'], axis=1, inplace=True)


categorical_cols = train_df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    train_df[col] = le.fit_transform(train_df[col])

# Standardize the numerical features
scaler = StandardScaler()
X = scaler.fit_transform(train_df)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Step 5: Tune AdaBoost parameters using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1, 2],
    'algorithm': ['SAMME', 'SAMME.R']
}
grid_search = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters from GridSearchCV:")
print(grid_search.best_params_)

# Step 6: Train AdaBoost model with best parameters
adaboost = grid_search.best_estimator_
adaboost.fit(X_train, y_train)

# Step 7: Cross-validation for model robustness
cv_scores = cross_val_score(adaboost, X_train, y_train, cv=5, scoring='accuracy')
print("\nCross-Validation Scores:")
print(cv_scores)
print(f"Mean Cross-Validation Score: {np.mean(cv_scores):.2f}")

# Step 8: Save the model
model_path = 'adaboost_model.pkl'
joblib.dump(adaboost, model_path)

# Step 9: Make predictions
y_pred = adaboost.predict(X_test)

# Step 10: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Additional metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.33)