import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import scipy.sparse
import time


# Load the CSV file into a DataFrame
file_path = '/Users/doa_ai/Developer/datamining/movie/data_raw.csv'
data = pd.read_csv(file_path)

# Create a folder to save distribution plots
output_folder = 'distribution'
os.makedirs(output_folder, exist_ok=True)

# Categorize columns into numerical and categorical
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = data.select_dtypes(include=['object', 'bool']).columns

# Reduce high-cardinality categorical features by grouping less frequent categories as 'Other'
for column in categorical_columns:
    if data[column].nunique() > 50:  # Threshold for high-cardinality categorical features
        top_categories = data[column].value_counts().nlargest(50).index
        data[column] = data[column].apply(lambda x: x if x in top_categories else 'Other')

# Group similar categories together (example: group genres or countries that share similar characteristics)
# For demonstration, we can use the genres feature to group action-related genres together
if 'genres' in categorical_columns:
    data['genres'] = data['genres'].replace({
        'Action': 'Action/Adventure',
        'Adventure': 'Action/Adventure',
        'Sci-Fi': 'Action/Adventure',
        'Comedy': 'Comedy/Drama',
        'Drama': 'Comedy/Drama'
    })

# Display the columns categorized as numerical and categorical
print(f"Numerical Columns: {numerical_columns.tolist()}")
print(f"Categorical Columns: {categorical_columns.tolist()}")

# Data Cleaning and Preparation using ColumnTransformer
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', num_imputer),
            ('scaler', StandardScaler())
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

# Feature Selection using SelectKBest with f_regression
k_best = SelectKBest(score_func=f_regression, k=5)
numerical_data_selected = k_best.fit_transform(numerical_data, numerical_data['vote_average'])  # Assuming 'vote_average' is the target column
selected_features = numerical_columns[k_best.get_support(indices=True)]
print(f"Selected Numerical Features using SelectKBest: {selected_features}")

# Feature Importance using Random Forest Regressor
forest = RandomForestRegressor(n_estimators=100, random_state=42)
forest.fit(numerical_data, numerical_data['vote_average'])
feature_importances_rf = pd.Series(forest.feature_importances_, index=numerical_columns)
important_features = feature_importances_rf.nlargest(5).index
print(f"Top 5 Important Features using RandomForest: {important_features}")

# Plot feature importance for RandomForest
plt.figure(figsize=(10, 6))
feature_importances_rf.sort_values(ascending=False).plot(kind='bar')
plt.title('Feature Importance using RandomForest')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'randomforest_feature_importance.png'))
plt.close()

# Interactive Visualization: Feature importance for RandomForest using Plotly
fig = px.bar(feature_importances_rf.sort_values(ascending=False).reset_index(), x='index', y=0, 
             labels={'index': 'Features', '0': 'Importance'}, 
             title='Feature Importance using RandomForest')
fig.update_layout(xaxis_title='Features', yaxis_title='Importance')
pio.write_html(fig, file=os.path.join(output_folder, 'randomforest_feature_importance_interactive.html'), auto_open=False)

# Feature Selection using L1 Regularization (Lasso)
lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(numerical_data, numerical_data['vote_average'])
l1_selector = SelectFromModel(lasso, prefit=True)
l1_selected_features = numerical_columns[l1_selector.get_support(indices=True)]
print(f"Selected Features using L1 Regularization (Lasso): {l1_selected_features}")

# Plot feature importance for Lasso
lasso_importance = pd.Series(abs(lasso.coef_), index=numerical_columns)
lasso_importance = lasso_importance[lasso_importance > 0]
plt.figure(figsize=(10, 6))
lasso_importance.sort_values(ascending=False).plot(kind='bar')
plt.title('Feature Importance using L1 Regularization (Lasso)')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'lasso_feature_importance.png'))
plt.close()

# Interactive Visualization: Feature importance for Lasso using Plotly
fig = px.bar(lasso_importance.sort_values(ascending=False).reset_index(), x='index', y=0, 
             labels={'index': 'Features', '0': 'Importance'}, 
             title='Feature Importance using L1 Regularization (Lasso)')
fig.update_layout(xaxis_title='Features', yaxis_title='Importance')
pio.write_html(fig, file=os.path.join(output_folder, 'lasso_feature_importance_interactive.html'), auto_open=False)

# Generate a comparison graph for feature importance from RandomForest and Lasso
importance_df = pd.DataFrame({
    'RandomForest': feature_importances_rf,
    'Lasso': lasso_importance
}).fillna(0)

plt.figure(figsize=(14, 8))
importance_df.sort_values(by='RandomForest', ascending=False).plot(kind='bar', width=0.8, ax=plt.gca())
plt.title('Feature Importance Comparison: RandomForest vs Lasso')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'feature_importance_comparison.png'))
plt.close()

# Interactive Visualization: Feature importance comparison using Plotly
importance_df_reset = importance_df.reset_index().melt(id_vars='index', var_name='Model', value_name='Importance')
fig = px.bar(importance_df_reset, x='index', y='Importance', color='Model', barmode='group',
             labels={'index': 'Features'}, title='Feature Importance Comparison: RandomForest vs Lasso')
fig.update_layout(xaxis_title='Features', yaxis_title='Importance')
pio.write_html(fig, file=os.path.join(output_folder, 'feature_importance_comparison_interactive.html'), auto_open=False)

# Update the numerical data with selected features
selected_features_combined = selected_features.union(important_features).union(l1_selected_features)
numerical_data = numerical_data[selected_features_combined]

# Re-analyze the cleaned data with selected features
print("Cleaned Data with Selected Features for Analysis:")
print(numerical_data.head())

# Visualization: Correlation matrix for selected numerical columns
if len(numerical_data.columns) > 1:
    correlation_matrix = numerical_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Selected Numerical Features')
    plt.savefig(os.path.join(output_folder, 'correlation_matrix_selected_features.png'))
    plt.close()

# Interactive Visualization: Correlation matrix using Plotly for selected numerical columns
if len(numerical_data.columns) > 1:
    fig = px.imshow(correlation_matrix, title='Interactive Correlation Matrix of Selected Numerical Features', color_continuous_scale='viridis')
    pio.write_html(fig, file=os.path.join(output_folder, 'correlation_matrix_selected_features_interactive.html'), auto_open=False)

# Summary statistics for selected dataset
summary_stats = numerical_data.describe()
print("Summary Statistics for Selected Features Dataset:")
print(summary_stats)

# Save summary statistics to a CSV file
summary_stats.to_csv(os.path.join(output_folder, 'summary_statistics_selected_features.csv'))

# Additional Visualizations for selected features
# Visualization: Pairplot for selected numerical columns
g = sns.pairplot(numerical_data, diag_kind='kde', plot_kws={'alpha': 0.5})
g.fig.suptitle('Pairplot of Selected Numerical Features', y=1.02)
g.savefig(os.path.join(output_folder, 'pairplot_selected_features.png'))
plt.close()

# Group similar visualizations (e.g., histograms) into one image
# Visualization: Combined histogram for selected numerical columns
plt.figure(figsize=(14, 10))
for column in numerical_data.columns:
    sns.histplot(numerical_data[column], kde=True, bins=30, label=column, alpha=0.5)
plt.title('Combined Distribution of Selected Numerical Features')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(os.path.join(output_folder, 'combined_histogram_selected_features.png'))
plt.close()

# Interactive Visualization: Pairplot using Plotly for selected numerical columns
if len(numerical_data.columns) > 1:
    fig = px.scatter_matrix(numerical_data, title='Scatter Matrix of Selected Numerical Features')
    pio.write_html(fig, file=os.path.join(output_folder, 'pairplot_selected_features_interactive.html'), auto_open=False)

# Interactive Visualization: Combined Histogram using Plotly
fig = go.Figure()
for column in numerical_data.columns:
    fig.add_trace(go.Histogram(x=numerical_data[column], name=column, opacity=0.5))
fig.update_layout(
    title='Combined Interactive Distribution of Selected Numerical Features',
    xaxis_title='Value',
    yaxis_title='Frequency',
    barmode='overlay'
)
pio.write_html(fig, file=os.path.join(output_folder, 'combined_histogram_selected_features_interactive.html'), auto_open=False)

# Boxplot for each selected numerical feature
g = sns.boxplot(data=numerical_data)
plt.title('Boxplot of Selected Numerical Features')
plt.xlabel('Features')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_folder, 'combined_boxplot_selected_features.png'))
plt.close()

# Interactive Visualization: Boxplot for each selected numerical feature using Plotly
fig = go.Figure()
for column in numerical_data.columns:
    fig.add_trace(go.Box(y=numerical_data[column], name=column))
fig.update_layout(
    title='Interactive Boxplot of Selected Numerical Features',
    xaxis_title='Features',
    yaxis_title='Value'
)
pio.write_html(fig, file=os.path.join(output_folder, 'combined_boxplot_selected_features_interactive.html'), auto_open=False)

# PCA Visualization for Selected Features
if len(numerical_data.columns) > 1:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(numerical_data)
    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])

    # Scatter plot for PCA components
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', data=pca_df)
    plt.title('PCA Visualization of Selected Numerical Features')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.savefig(os.path.join(output_folder, 'pca_scatterplot_selected_features.png'))
    plt.close()

    # Explained variance plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.title('PCA Variance Contribution for Selected Features')
    plt.savefig(os.path.join(output_folder, 'pca_variance_contribution_selected_features.png'))
    plt.close()

    # Interactive PCA Visualization using Plotly
    fig = px.scatter(pca_df, x='PCA1', y='PCA2', title='Interactive PCA Visualization of Selected Numerical Features')
    fig.update_layout(xaxis_title='PCA1', yaxis_title='PCA2')
    pio.write_html(fig, file=os.path.join(output_folder, 'pca_scatterplot_selected_features_interactive.html'), auto_open=False)

# t-SNE Visualization for Selected Features
if len(numerical_data.columns) > 1:
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(numerical_data)
    tsne_df = pd.DataFrame(data=tsne_result, columns=['TSNE1', 'TSNE2'])

    # Scatter plot for t-SNE components
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TSNE1', y='TSNE2', data=tsne_df)
    plt.title('t-SNE Visualization of Selected Numerical Features')
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.savefig(os.path.join(output_folder, 'tsne_scatterplot_selected_features.png'))
    plt.close()

    # Interactive t-SNE Visualization using Plotly
    fig = px.scatter(tsne_df, x='TSNE1', y='TSNE2', title='Interactive t-SNE Visualization of Selected Numerical Features')
    fig.update_layout(xaxis_title='TSNE1', yaxis_title='TSNE2')
    pio.write_html(fig, file=os.path.join(output_folder, 'tsne_scatterplot_selected_features_interactive.html'), auto_open=False)

# Comparison between PCA and t-SNE Visualizations
if len(numerical_data.columns) > 1:
    combined_df = pd.concat([pca_df, tsne_df], axis=1)
    # Ensure the Method column matches the length of combined_df correctly
    combined_df = pd.concat([pca_df, tsne_df], ignore_index=True)
    combined_df['Method'] = ['PCA'] * len(pca_df) + ['t-SNE'] * len(tsne_df)


    # Plotly comparison visualization for PCA and t-SNE
    fig = px.scatter(combined_df, x='PCA1', y='PCA2', color='Method', title='Comparison of PCA and t-SNE Visualizations',
                     labels={'PCA1': 'PCA / TSNE Dimension 1', 'PCA2': 'PCA / TSNE Dimension 2'})
    fig.add_trace(go.Scatter(x=combined_df['TSNE1'], y=combined_df['TSNE2'], mode='markers', name='t-SNE'))
    pio.write_html(fig, file=os.path.join(output_folder, 'pca_tsne_comparison_interactive.html'), auto_open=False)

# Visualize clusters from t-SNE and PCA
if len(numerical_data.columns) > 1:
    from sklearn.cluster import KMeans

    # Apply KMeans clustering
    kmeans_pca = KMeans(n_clusters=3, random_state=42).fit(pca_df)
    pca_df['Cluster'] = kmeans_pca.labels_

    kmeans_tsne = KMeans(n_clusters=3, random_state=42).fit(tsne_df)
    tsne_df['Cluster'] = kmeans_tsne.labels_

    # Plot clusters for PCA
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='viridis', data=pca_df)
    plt.title('PCA Clusters Visualization')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.savefig(os.path.join(output_folder, 'pca_clusters_visualization.png'))
    plt.close()

    # Plot clusters for t-SNE
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster', palette='viridis', data=tsne_df)
    plt.title('t-SNE Clusters Visualization')
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.savefig(os.path.join(output_folder, 'tsne_clusters_visualization.png'))
    plt.close()

    # Interactive cluster visualization for PCA using Plotly
    fig = px.scatter(pca_df, x='PCA1', y='PCA2', color='Cluster', title='Interactive PCA Clusters Visualization')
    fig.update_layout(xaxis_title='PCA1', yaxis_title='PCA2')
    pio.write_html(fig, file=os.path.join(output_folder, 'pca_clusters_visualization_interactive.html'), auto_open=False)

    # Interactive cluster visualization for t-SNE using Plotly
    fig = px.scatter(tsne_df, x='TSNE1', y='TSNE2', color='Cluster', title='Interactive t-SNE Clusters Visualization')
    fig.update_layout(xaxis_title='TSNE1', yaxis_title='TSNE2')
    pio.write_html(fig, file=os.path.join(output_folder, 'tsne_clusters_visualization_interactive.html'), auto_open=False)

# Compare computation costs of PCA vs t-SNE
if len(numerical_data.columns) > 1:
    start_time = time.time()
    PCA(n_components=2).fit_transform(numerical_data)
    pca_time = time.time() - start_time

    start_time = time.time()
    TSNE(n_components=2, random_state=42).fit_transform(numerical_data)
    tsne_time = time.time() - start_time

    # Visualization of computation times
    time_df = pd.DataFrame({
        'Method': ['PCA', 't-SNE'],
        'Computation Time (s)': [pca_time, tsne_time]
    })

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Method', y='Computation Time (s)', data=time_df, palette='viridis')
    plt.title('Computation Time Comparison: PCA vs t-SNE')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Method')
    plt.savefig(os.path.join(output_folder, 'pca_tsne_computation_time_comparison.png'))
    plt.close()

    # Interactive computation time visualization using Plotly
    fig = px.bar(time_df, x='Method', y='Computation Time (s)', title='Interactive Computation Time Comparison: PCA vs t-SNE')
    fig.update_layout(xaxis_title='Method', yaxis_title='Computation Time (seconds)')
    pio.write_html(fig, file=os.path.join(output_folder, 'pca_tsne_computation_time_comparison_interactive.html'), auto_open=False)
