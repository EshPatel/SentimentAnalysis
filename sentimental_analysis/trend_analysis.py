import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load YouTube comments dataset
file_path = 'scrapers/csv_outputs/youtube_data.csv'  # Change path if needed
df = pd.read_csv(file_path)

# Ensure the 'Text' column exists
if 'Text' not in df.columns:
    raise ValueError("Dataset must have a 'Text' column containing the comments.")

# Preprocessing Function
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = re.sub(r'http\S+|[^\w\s]', ' ', str(text))  # Remove URLs and special characters
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Apply preprocessing
df['cleaned_text'] = df['Text'].apply(preprocess_text)

# Convert text into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_text'])

# 🔥 **1️⃣ Determine Optimal Number of Clusters Using Elbow Method**
wcss = []
silhouette_scores = []
K_range = range(5, 15)  # Trying different cluster numbers

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # Inertia = WCSS
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))  # Measure clustering quality

# **Plot Elbow Curve**
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

# **Plot Silhouette Score Curve**
plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o', linestyle='-', color='g')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.show()

# ✅ **Choose the Best Number of Clusters Based on the Elbow Point**
optimal_clusters = 13  # Change this after checking the elbow curve

# 🔥 **2️⃣ Apply K-Means with Optimal Clusters**
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

# **Trending Topics Extraction**
trending_topics = {}
for cluster_num in range(optimal_clusters):
    cluster_comments = df[df['cluster'] == cluster_num]['cleaned_text']
    all_words = ' '.join(cluster_comments).split()
    trending_topics[cluster_num] = Counter(all_words).most_common(10)

# Print trending topics
print("\n🔍 **Trending Topics (Top Words per Cluster):**")
for cluster, words in trending_topics.items():
    print(f"\nCluster {cluster}: {words}")

# 📊 **3️⃣ Visualizing Trending Topics**
plt.figure(figsize=(10, 5))
sns.countplot(x=df['cluster'], palette='viridis')
plt.xlabel('Topic Cluster')
plt.ylabel('Number of Comments')
plt.title('Trending Topics Distribution')
plt.show()

# 🌥️ **4️⃣ Generate Word Cloud for Most Popular Topic**
most_popular_cluster = df['cluster'].value_counts().idxmax()
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(
    ' '.join(df[df['cluster'] == most_popular_cluster]['cleaned_text'])
)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(f"Trending Words in Cluster {most_popular_cluster}")
plt.show()

# 📁 Save analyzed data
output_file = 'trend_analysis_youtube_data_ml.csv'
df.to_csv(output_file, index=False)

print("\nAnalysis Complete!")
