import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

csv_path = '/Users/doa_ai/Developer/datamining/movie/data_raw.csv' # pass path of dataset here
data = pd.read_csv(csv_path)

# Select relevant numerical columns for visualization
numerical_cols = ['vote_average', 'vote_count', 'revenue', 'runtime', 'averageRating', 'numVotes']

# Convert the 'release_date' column to datetime format to extract year information
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')

# Extract the release year from the 'release_date' column
data['release_year'] = data['release_date'].dt.year

# Group data by release year and calculate average revenue, vote average, and vote count
yearly_stats = data.groupby('release_year').agg({
    'revenue': 'mean',
    'vote_average': 'mean',
    'vote_count': 'mean'
}).dropna().reset_index()

# Set up the visual style
sns.set_theme(style="whitegrid")

# Create a pairplot to explore relationships between the numerical columns
plt.figure(figsize=(16, 10))
sns.pairplot(data[numerical_cols], diag_kind="kde", plot_kws={"alpha": 0.5})
plt.suptitle('Pairwise Relationships Between Key Movie Attributes', y=1.02)
plt.savefig('./distribution/pairwise_relationships.png')
plt.close()

# Plot distribution of 'vote_average' and 'averageRating' to understand rating patterns
plt.figure(figsize=(14, 6))
sns.histplot(data['vote_average'], kde=True, color='blue', label='Vote Average', bins=20)
sns.histplot(data['averageRating'], kde=True, color='orange', label='Average Rating', bins=20)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Vote Average and Average Rating')
plt.legend()
plt.savefig('./distribution/distribution_vote_average_rating.png')
plt.close()

# Create bar plot of average revenue for each director
avg_revenue_director = data.groupby('directors')['revenue'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_revenue_director.values, y=avg_revenue_director.index, palette="viridis")
plt.xlabel('Average Revenue (in billions)')
plt.title('Top 10 Directors by Average Movie Revenue')
plt.savefig('./distribution/top_directors_revenue.png')
plt.close()

# Plot the relationship between runtime and revenue
plt.figure(figsize=(10, 6))
sns.scatterplot(x='runtime', y='revenue', data=data, alpha=0.7)
plt.xlabel('Runtime (minutes)')
plt.ylabel('Revenue (in billions)')
plt.title('Movie Runtime vs Revenue')
plt.savefig('./distribution/runtime_vs_revenue.png')
plt.close()

# Plot vote count vs. vote average to see voting patterns
plt.figure(figsize=(10, 6))
sns.scatterplot(x='vote_count', y='vote_average', data=data, alpha=0.7, color='red')
plt.xlabel('Vote Count')
plt.ylabel('Vote Average')
plt.title('Vote Count vs Vote Average')
plt.savefig('./distribution/vote_count_vs_average.png')
plt.close()

# Plot top genres by number of movies
data['genres_split'] = data['genres'].str.split(', ')
genres_exploded = data.explode('genres_split')
top_genres = genres_exploded['genres_split'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, palette='rocket')
plt.xlabel('Number of Movies')
plt.title('Top 10 Movie Genres by Number of Movies')
plt.savefig('./distribution/top_genres.png')
plt.close()

# Clean the numerical columns by converting them to numeric and handling errors
for col in numerical_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with NaN values in these columns
cleaned_data = data.dropna(subset=numerical_cols)

# Create the pairplot again with cleaned data
plt.figure(figsize=(16, 10))
sns.pairplot(cleaned_data[numerical_cols], diag_kind="kde", plot_kws={"alpha": 0.5})
plt.suptitle('Pairwise Relationships Between Key Movie Attributes (Cleaned Data)', y=1.02)
plt.savefig('./distribution/pairwise_relationships_cleaned.png')
plt.close()

# Plot distribution of 'vote_average' and 'averageRating' with cleaned data
plt.figure(figsize=(14, 6))
sns.histplot(cleaned_data['vote_average'], kde=True, color='blue', label='Vote Average', bins=20)
sns.histplot(cleaned_data['averageRating'], kde=True, color='orange', label='Average Rating', bins=20)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Vote Average and Average Rating (Cleaned Data)')
plt.legend()
plt.savefig('./distribution/distribution_vote_average_rating_cleaned.png')
plt.close()

# Create bar plot of average revenue for each director (cleaned data)
avg_revenue_director_cleaned = cleaned_data.groupby('directors')['revenue'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_revenue_director_cleaned.values, y=avg_revenue_director_cleaned.index, palette="viridis")
plt.xlabel('Average Revenue (in billions)')
plt.title('Top 10 Directors by Average Movie Revenue (Cleaned Data)')
plt.savefig('./distribution/top_directors_revenue_cleaned.png')
plt.close()

# Plot the relationship between runtime and revenue (cleaned data)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='runtime', y='revenue', data=cleaned_data, alpha=0.7)
plt.xlabel('Runtime (minutes)')
plt.ylabel('Revenue (in billions)')
plt.title('Movie Runtime vs Revenue (Cleaned Data)')
plt.savefig('./distribution/runtime_vs_revenue_cleaned.png')
plt.close()

# Plot vote count vs. vote average to see voting patterns (cleaned data)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='vote_count', y='vote_average', data=cleaned_data, alpha=0.7, color='red')
plt.xlabel('Vote Count')
plt.ylabel('Vote Average')
plt.title('Vote Count vs Vote Average (Cleaned Data)')
plt.savefig('./distribution/vote_count_vs_average_cleaned.png')
plt.close()

# Plot top genres by number of movies (cleaned data)
cleaned_data['genres_split'] = cleaned_data['genres'].str.split(', ')
genres_exploded_cleaned = cleaned_data.explode('genres_split')
top_genres_cleaned = genres_exploded_cleaned['genres_split'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_genres_cleaned.values, y=top_genres_cleaned.index, palette='rocket')
plt.xlabel('Number of Movies')
plt.title('Top 10 Movie Genres by Number of Movies (Cleaned Data)')
plt.savefig('./distribution/top_genres_cleaned.png')
plt.close()

# Further clean numerical columns to ensure they are purely numeric and drop any non-numeric rows
cleaned_data = cleaned_data[numerical_cols].dropna().astype(float)

# Create histograms to explore distributions of numerical columns individually
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Distributions of Key Numerical Movie Attributes', fontsize=16)

# Plot each numerical column
for ax, col in zip(axes.flat, numerical_cols):
    sns.histplot(cleaned_data[col], kde=True, ax=ax, bins=20)
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('./distribution/numerical_distributions.png')
plt.close()

# Re-attempt scatter plots for relationships between key metrics
plt.figure(figsize=(10, 6))
sns.scatterplot(x='runtime', y='revenue', data=cleaned_data, alpha=0.7)
plt.xlabel('Runtime (minutes)')
plt.ylabel('Revenue (in billions)')
plt.title('Movie Runtime vs Revenue (Cleaned Data)')
plt.savefig('./distribution/runtime_vs_revenue_final.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='vote_count', y='vote_average', data=cleaned_data, alpha=0.7, color='red')
plt.xlabel('Vote Count')
plt.ylabel('Vote Average')
plt.title('Vote Count vs Vote Average (Cleaned Data)')
plt.savefig('./distribution/vote_count_vs_average_final.png')
plt.close()

# Save the plots showing yearly trends as images
plt.figure(figsize=(14, 6))
sns.lineplot(x='release_year', y='revenue', data=yearly_stats, marker='o')
plt.xlabel('Release Year')
plt.ylabel('Average Revenue (in billions)')
plt.title('Average Movie Revenue Over the Years')
plt.savefig('./distribution/average_revenue_over_years.png')
plt.close()

plt.figure(figsize=(14, 6))
sns.lineplot(x='release_year', y='vote_average', data=yearly_stats, marker='o', color='green')
plt.xlabel('Release Year')
plt.ylabel('Average Vote Average')
plt.title('Average Movie Rating Over the Years')
plt.savefig('./distribution/average_rating_over_years.png')
plt.close()

plt.figure(figsize=(14, 6))
sns.lineplot(x='release_year', y='vote_count', data=yearly_stats, marker='o', color='red')
plt.xlabel('Release Year')
plt.ylabel('Average Vote Count')
plt.title('Average Vote Count Over the Years')
plt.savefig('./distribution/average_vote_count_over_years.png')
plt.close()

# Additional visualizations

# Revenue by Genre
avg_revenue_genre = genres_exploded.groupby('genres_split')['revenue'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_revenue_genre.values, y=avg_revenue_genre.index, palette='viridis')
plt.xlabel('Average Revenue (in billions)')
plt.title('Top 10 Genres by Average Revenue')
plt.savefig('./distribution/top_genres_revenue.png')
plt.close()

# Popularity Over Time
genres_count_by_year = genres_exploded.groupby(['release_year', 'genres_split']).size().reset_index(name='count')
plt.figure(figsize=(14, 8))
sns.lineplot(data=genres_count_by_year, x='release_year', y='count', hue='genres_split', estimator='sum', ci=None)
plt.xlabel('Release Year')
plt.ylabel('Number of Movies')
plt.title('Number of Movies Released by Genre Over Time')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('./distribution/genres_popularity_over_time.png')
plt.close()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
corr_matrix = cleaned_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.savefig('./distribution/correlation_heatmap.png')
plt.close()

# Revenue Distribution by Runtime
cleaned_data['runtime_category'] = pd.cut(cleaned_data['runtime'], bins=[0, 90, 120, 150, 300], labels=['Short', 'Medium', 'Long', 'Very Long'])
plt.figure(figsize=(12, 6))
sns.boxplot(x='runtime_category', y='revenue', data=cleaned_data, palette='Set2')
plt.xlabel('Runtime Category')
plt.ylabel('Revenue (in billions)')
plt.title('Revenue Distribution by Runtime Category')
plt.savefig('./distribution/revenue_by_runtime_category.png')
plt.close()

# Director Analysis (Number of Movies, Average Rating, Average Revenue)
top_directors = genres_exploded['directors'].value_counts().head(10).index
director_stats = data[data['directors'].isin(top_directors)].groupby('directors').agg({
    'revenue': 'mean',
    'vote_average': 'mean',
    'id': 'count'
}).rename(columns={'id': 'movie_count'}).sort_values(by='movie_count', ascending=False).reset_index()

fig, axes = plt.subplots(3, 1, figsize=(12, 18))
fig.suptitle('Top Directors Analysis', fontsize=16)

sns.barplot(x=director_stats['movie_count'], y=director_stats['directors'], ax=axes[0], palette='Blues')
axes[0].set_title('Number of Movies by Director')
axes[0].set_xlabel('Number of Movies')
axes[0].set_ylabel('Director')

sns.barplot(x=director_stats['revenue'], y=director_stats['directors'], ax=axes[1], palette='Greens')
axes[1].set_title('Average Revenue by Director')
axes[1].set_xlabel('Average Revenue (in billions)')
axes[1].set_ylabel('Director')

sns.barplot(x=director_stats['vote_average'], y=director_stats['directors'], ax=axes[2], palette='Oranges')
axes[2].set_title('Average Vote Rating by Director')
axes[2].set_xlabel('Average Vote Rating')
axes[2].set_ylabel('Director')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('./distribution/director_analysis.png')
plt.close()

# Additional visualizations to demonstrate reasons for choosing average rating and vote average as target labels

# Average Rating vs Revenue
plt.figure(figsize=(10, 6))
sns.scatterplot(x='averageRating', y='revenue', data=cleaned_data, alpha=0.7, color='purple')
plt.xlabel('Average Rating')
plt.ylabel('Revenue (in billions)')
plt.title('Relationship Between Average Rating and Revenue')
plt.savefig('./distribution/average_rating_vs_revenue.png')
plt.close()

# Vote Average vs Revenue
plt.figure(figsize=(10, 6))
sns.scatterplot(x='vote_average', y='revenue', data=cleaned_data, alpha=0.7, color='blue')
plt.xlabel('Vote Average')
plt.ylabel('Revenue (in billions)')
plt.title('Relationship Between Vote Average and Revenue')
plt.savefig('./distribution/vote_average_vs_revenue.png')
plt.close()

# Average Rating and Vote Average Distribution by Genre
avg_rating_genre = genres_exploded.groupby('genres_split')['averageRating'].mean().sort_values(ascending=False)
vote_avg_genre = genres_exploded.groupby('genres_split')['vote_average'].mean().sort_values(ascending=False)

plt.figure(figsize=(14, 6))
sns.barplot(x=avg_rating_genre.values, y=avg_rating_genre.index, palette='viridis', label='Average Rating')
sns.barplot(x=vote_avg_genre.values, y=vote_avg_genre.index, palette='rocket', alpha=0.5, label='Vote Average')
plt.xlabel('Rating')
plt.ylabel('Genre')
plt.title('Average Rating and Vote Average by Genre')
plt.legend()
plt.savefig('./distribution/average_rating_vote_average_by_genre.png')
plt.close()

# Correlation Heatmap Focused on Target Labels
plt.figure(figsize=(10, 8))
target_corr_matrix = cleaned_data[['averageRating', 'vote_average', 'revenue', 'runtime', 'vote_count', 'numVotes']].corr()
sns.heatmap(target_corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap Focused on Target Labels (Average Rating & Vote Average)')
plt.savefig('./distribution/target_labels_correlation_heatmap.png')
plt.close()

# Additional visualizations for categorical columns

# Movie Count by Certification
plt.figure(figsize=(12, 6))
certification_count = data['certification'].value_counts().head(10)
sns.barplot(x=certification_count.values, y=certification_count.index, palette='magma')
plt.xlabel('Number of Movies')
plt.ylabel('Certification')
plt.title('Top 10 Movie Certifications by Number of Movies')
plt.savefig('./distribution/movie_count_by_certification.png')
plt.close()

# Movie Count by Production Company
plt.figure(figsize=(12, 6))
production_company_count = data['production_companies'].value_counts().head(10)
sns.barplot(x=production_company_count.values, y=production_company_count.index, palette='cool')
plt.xlabel('Number of Movies')
plt.ylabel('Production Company')
plt.title('Top 10 Production Companies by Number of Movies')
plt.savefig('./distribution/movie_count_by_production_company.png')
plt.close()

# Movie Count by Country
plt.figure(figsize=(12, 6))
country_count = data['production_countries'].value_counts().head(10)
sns.barplot(x=country_count.values, y=country_count.index, palette='viridis')
plt.xlabel('Number of Movies')
plt.ylabel('Country')
plt.title('Top 10 Countries by Number of Movies Produced')
plt.savefig('./distribution/movie_count_by_country.png')
plt.close()

# Movie Count by Language
plt.figure(figsize=(12, 6))
language_count = data['original_language'].value_counts().head(10)
sns.barplot(x=language_count.values, y=language_count.index, palette='plasma')
plt.xlabel('Number of Movies')
plt.ylabel('Original Language')
plt.title('Top 10 Original Languages by Number of Movies')
plt.savefig('./distribution/movie_count_by_language.png')
plt.close()
