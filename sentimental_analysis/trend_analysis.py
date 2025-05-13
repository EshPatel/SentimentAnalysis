import pandas as pd
import numpy as np
import re
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
from datetime import datetime
import networkx as nx

ANALYZED_DATA_PATH = 'scrapers/csv_outputs/analyzed_youtube_data.csv'
TIMESTAMP_COLUMN = 'PublishedAt'
OUTPUT_DIR = "presentation_outputs_advanced_trends"
TREND_ANALYSIS_CSV_OUTPUT = 'advanced_trend_analysis_predictions.csv'
TOP_N_CLUSTERS_FOR_DEEP_DIVE = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

nltk_resources = {'stopwords': 'stopwords', 'punkt': 'punkt'}
for resource, package in nltk_resources.items():
    try: nltk.data.find(f'corpora/{resource}')
    except: nltk.download(package, quiet=True)
stop_words_list = stopwords.words('english')

print(f"--- [trend_analysis_advanced.py] Advanced Trend Analysis Started ---")
try:
    df = pd.read_csv(ANALYZED_DATA_PATH)
    print(f"[trend_analysis_advanced.py] Loaded {len(df)} rows from {ANALYZED_DATA_PATH}")
except FileNotFoundError:
    print(f"[trend_analysis_advanced.py] ERROR: File not found: {ANALYZED_DATA_PATH}")
    exit()

required_cols = ['cleaned_text', 'sentiment_score', 'sentiment_category']
if TIMESTAMP_COLUMN: required_cols.append(TIMESTAMP_COLUMN)
missing_cols = [col for col in required_cols if col not in df.columns and col != TIMESTAMP_COLUMN]
if missing_cols:
    print(f"[trend_analysis_advanced.py] ERROR: Missing required columns: {missing_cols}")
    exit()

has_timestamp = TIMESTAMP_COLUMN and TIMESTAMP_COLUMN in df.columns
if TIMESTAMP_COLUMN and not has_timestamp:
    print(f"[trend_analysis_advanced.py] WARNING: Timestamp column '{TIMESTAMP_COLUMN}' not found. Temporal analysis limited.")

df.dropna(subset=['cleaned_text'], inplace=True)
if df.empty:
    print("[trend_analysis_advanced.py] ERROR: No valid text data after dropping NaNs.")
    exit()
print(f"[trend_analysis_advanced.py] Data ready: {len(df)} rows.")

print("[trend_analysis_advanced.py] Vectorizing text (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=2000, stop_words=stop_words_list, ngram_range=(1,2))
X = vectorizer.fit_transform(df['cleaned_text'])
print(f"[trend_analysis_advanced.py] TF-IDF matrix shape: {X.shape}")
feature_names = vectorizer.get_feature_names_out()

print("[trend_analysis_advanced.py] Determining optimal number of clusters (5-15)...")
wcss = []
silhouette_scores = []
K_range = range(5, 16)
for k_val in K_range:
    km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    km.fit(X)
    wcss.append(km.inertia_)
    if X.shape[0] > 1 and X.shape[0] <= 5000:
        try: silhouette_scores.append(silhouette_score(X, km.labels_))
        except: silhouette_scores.append(np.nan)
    else: silhouette_scores.append(np.nan)

plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o', color='b')
plt.title('Elbow Method for Optimal k'); plt.xlabel('Number of Clusters (k)'); plt.ylabel('WCSS (Inertia)')
plt.xticks(K_range); plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'adv_elbow_plot.png')); plt.close()
print(f"[trend_analysis_advanced.py] Saved plot: adv_elbow_plot.png")

if any(not np.isnan(s) for s in silhouette_scores):
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, silhouette_scores, marker='o', color='g')
    plt.title('Silhouette Score vs. Number of Clusters'); plt.xlabel('Number of Clusters (k)'); plt.ylabel('Silhouette Score')
    plt.xticks(K_range); plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'adv_silhouette_plot.png')); plt.close()
    print(f"[trend_analysis_advanced.py] Saved plot: adv_silhouette_plot.png")
    try:
        valid_scores = [s for s in silhouette_scores if not np.isnan(s)]
        if valid_scores:
            optimal_k_silhouette = K_range.start + np.nanargmax(silhouette_scores)
            print(f"[trend_analysis_advanced.py] Suggestion: Optimal k (Silhouette) ~ {optimal_k_silhouette}")
    except: pass

optimal_clusters = 5
print(f"[trend_analysis_advanced.py] Using k = {optimal_clusters} clusters.")

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
order_centroids = centroids.argsort()[:, ::-1]

print("[trend_analysis_advanced.py] Analyzing cluster characteristics...")
cluster_analysis = []
terms_per_cluster_display = 10
for i in range(optimal_clusters):
    cluster_df = df[df['cluster'] == i]
    if cluster_df.empty: continue
    top_terms = [feature_names[ind] for ind in order_centroids[i, :terms_per_cluster_display]]
    cluster_info = {
        'Cluster': i, 'Size': len(cluster_df),
        'AvgSentiment': cluster_df['sentiment_score'].mean(),
        'PositiveRatio': cluster_df[cluster_df['sentiment_category'] == 'Positive'].shape[0] / len(cluster_df) if len(cluster_df) > 0 else 0,
        'TopTerms': ', '.join(top_terms)
    }
    cluster_analysis.append(cluster_info)
cluster_summary_df = pd.DataFrame(cluster_analysis)
cluster_summary_df['TrendScore'] = (np.log1p(cluster_summary_df['Size']) *
                                    (cluster_summary_df['AvgSentiment'] + 1.1) *
                                    (cluster_summary_df['PositiveRatio'] + 0.1))
min_score, max_score = cluster_summary_df['TrendScore'].min(), cluster_summary_df['TrendScore'].max()
cluster_summary_df['TrendScore_Normalized'] = 100 * (cluster_summary_df['TrendScore'] - min_score) / (max_score - min_score) if max_score > min_score else 50.0
cluster_summary_df = cluster_summary_df.sort_values(by='TrendScore_Normalized', ascending=False).reset_index(drop=True)


if has_timestamp:
    print(f"[trend_analysis_advanced.py] Performing temporal analysis using: {TIMESTAMP_COLUMN}...")
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors='coerce')
    df.dropna(subset=[TIMESTAMP_COLUMN], inplace=True)
    if not df.empty:
        df_sorted_time = df.sort_values(by=TIMESTAMP_COLUMN)
        df_sorted_time['TimePeriod'] = df_sorted_time[TIMESTAMP_COLUMN].dt.to_period('W')
        
        temporal_cluster_summary = df_sorted_time.groupby(['cluster', 'TimePeriod']).agg(
            PeriodSize=('sentiment_score', 'size'),
            PeriodAvgSentiment=('sentiment_score', 'mean')
        ).reset_index()

        for idx, row in cluster_summary_df.head(TOP_N_CLUSTERS_FOR_DEEP_DIVE).iterrows():
            cluster_id = row['Cluster']
            data_c = temporal_cluster_summary[temporal_cluster_summary['cluster'] == cluster_id]
            if len(data_c) > 1:
                fig, ax1 = plt.subplots(figsize=(12,6))
                time_p_str = data_c['TimePeriod'].astype(str)
                ax1.plot(time_p_str, data_c['PeriodSize'], color='tab:blue', marker='o', label='Size')
                ax1.set_xlabel('Time Period (Week)'); ax1.set_ylabel('Cluster Size', color='tab:blue')
                ax1.tick_params(axis='y', labelcolor='tab:blue'); ax1.tick_params(axis='x', rotation=45)
                ax2 = ax1.twinx()
                ax2.plot(time_p_str, data_c['PeriodAvgSentiment'], color='tab:red', marker='x', linestyle='--', label='Avg Sentiment')
                ax2.set_ylabel('Average Sentiment', color='tab:red'); ax2.tick_params(axis='y', labelcolor='tab:red')
                ax2.set_ylim(-1.1, 1.1)
                plt.title(f'Temporal Trend for Potential Trend Cluster {cluster_id}'); fig.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f'adv_temporal_trend_cluster_{cluster_id}.png')); plt.close()
                print(f"[trend_analysis_advanced.py] Saved temporal plot for Cluster {cluster_id}")

        print("[trend_analysis_advanced.py] Performing simplified keyword burst detection...")
        bursting_keywords = {}
        if len(df_sorted_time['TimePeriod'].unique()) >=2 :
            recent_period = df_sorted_time['TimePeriod'].unique()[-1]
            previous_period = df_sorted_time['TimePeriod'].unique()[-2]
            for i in range(TOP_N_CLUSTERS_FOR_DEEP_DIVE):
                cluster_id = cluster_summary_df.iloc[i]['Cluster']
                top_terms_for_cluster = cluster_summary_df.iloc[i]['TopTerms'].split(', ')[:5]
                
                cluster_texts_recent = df_sorted_time[
                    (df_sorted_time['cluster'] == cluster_id) & (df_sorted_time['TimePeriod'] == recent_period)
                ]['cleaned_text']
                cluster_texts_previous = df_sorted_time[
                    (df_sorted_time['cluster'] == cluster_id) & (df_sorted_time['TimePeriod'] == previous_period)
                ]['cleaned_text']

                if cluster_texts_recent.empty or cluster_texts_previous.empty: continue

                recent_counts = Counter(' '.join(cluster_texts_recent).split())
                previous_counts = Counter(' '.join(cluster_texts_previous).split())
                
                cluster_bursts = []
                for term in top_terms_for_cluster:
                    increase_factor = (recent_counts.get(term, 0) +1 ) / (previous_counts.get(term, 0) + 1)
                    if recent_counts.get(term, 0) > 5 and increase_factor > 2.0:
                        cluster_bursts.append(f"{term} (x{increase_factor:.1f})")
                if cluster_bursts:
                    bursting_keywords[cluster_id] = ", ".join(cluster_bursts)
            if bursting_keywords:
                 print(f"[trend_analysis_advanced.py] Potential bursting keywords in top clusters: {bursting_keywords}")
                 cluster_summary_df['BurstingKeywords'] = cluster_summary_df['Cluster'].map(bursting_keywords)

print("[trend_analysis_advanced.py] Performing keyword co-occurrence network analysis for top cluster...")
top_cluster_for_network = cluster_summary_df.iloc[0]['Cluster']
top_cluster_texts = df[df['cluster'] == top_cluster_for_network]['cleaned_text'].tolist()
tokenized_texts_for_network = [text.split() for text in top_cluster_texts]

cooccurrence_counts = Counter()
window_size = 5
for tokens in tokenized_texts_for_network:
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + window_size, len(tokens))):
            term1, term2 = sorted((tokens[i], tokens[j]))
            if term1 != term2:
                cooccurrence_counts[(term1, term2)] += 1

if cooccurrence_counts:
    G = nx.Graph()
    min_cooccurrence = 3
    for (term1, term2), count in cooccurrence_counts.items():
        if count >= min_cooccurrence:
            G.add_edge(term1, term2, weight=count)

    if G.number_of_nodes() > 0 and G.number_of_edges() > 0 :
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42) 
        
        node_sizes = [G.degree(node) * 100 for node in G.nodes()]
        edge_widths = [G.edges[edge]['weight'] * 0.5 for edge in G.edges()]

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        plt.title(f"Keyword Co-occurrence Network for Top Trend Cluster {top_cluster_for_network}")
        plt.axis('off')
        network_plot_path = os.path.join(OUTPUT_DIR, f'adv_network_cluster_{top_cluster_for_network}.png')
        plt.savefig(network_plot_path); plt.close()
        print(f"[trend_analysis_advanced.py] Saved keyword network plot: {network_plot_path}")
    else:
        print(f"[trend_analysis_advanced.py] Network for cluster {top_cluster_for_network} is too sparse to plot.")
else:
    print(f"[trend_analysis_advanced.py] No co-occurrences found for network analysis in cluster {top_cluster_for_network}.")


print("\n" + "="*50)
print(" Advanced Potential Trend Clusters (Ranked by Score) ")
print("="*50)
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.max_colwidth = 100 
print(cluster_summary_df.to_string(index=False))
pd.reset_option('display.float_format')
pd.reset_option('display.max_colwidth')
print("="*50 + "\n")

enriched_df_path = os.path.join(os.path.dirname(ANALYZED_DATA_PATH), TREND_ANALYSIS_CSV_OUTPUT)
df.to_csv(enriched_df_path, index=False)
print(f"[trend_analysis_advanced.py] Saved enriched DataFrame to: {enriched_df_path}")
cluster_summary_path = os.path.join(OUTPUT_DIR, 'adv_cluster_trend_summary.csv')
cluster_summary_df.to_csv(cluster_summary_path, index=False, float_format="%.4f")
print(f"[trend_analysis_advanced.py] Saved cluster summary to: {cluster_summary_path}")

plt.figure(figsize=(12, 7))
sns.barplot(x='Cluster', y='TrendScore_Normalized', data=cluster_summary_df.head(10),
            palette='coolwarm_r', order=cluster_summary_df.head(10)['Cluster'])
plt.title('Relative Trend Potential Score by Topic Cluster (Top 10)')
plt.xlabel('Topic Cluster ID'); plt.ylabel('Normalized Trend Potential Score (0-100)')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'adv_cluster_trend_score_comparison.png')); plt.close()
print(f"[trend_analysis_advanced.py] Saved plot: adv_cluster_trend_score_comparison.png")

plt.figure(figsize=(10, 7))
sns.scatterplot(x='Size', y='AvgSentiment', hue='Cluster', size='PositiveRatio',
    sizes=(50, 500), palette='viridis', data=cluster_summary_df, legend='brief')
plt.title('Cluster Sentiment Profile'); plt.xlabel('Cluster Size'); plt.ylabel('Average Sentiment Score')
plt.axhline(0, color='grey', linestyle='--', lw=0.8); plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(os.path.join(OUTPUT_DIR, 'adv_cluster_sentiment_profile.png')); plt.close()
print(f"[trend_analysis_advanced.py] Saved plot: adv_cluster_sentiment_profile.png")

print("[trend_analysis_advanced.py] Generating word clouds for top trend clusters...")
for i in range(min(TOP_N_CLUSTERS_FOR_DEEP_DIVE, len(cluster_summary_df))):
    row = cluster_summary_df.iloc[i]
    cluster_id = row['Cluster']
    cluster_text = ' '.join(df[df['cluster'] == cluster_id]['cleaned_text'].astype(str))
    if not cluster_text.strip(): continue
    try:
        wc = WordCloud(width=1000, height=500, background_color='white', colormap='viridis', max_words=75).generate(cluster_text)
        plt.figure(figsize=(10,5)); plt.imshow(wc, interpolation='bilinear'); plt.axis('off')
        plt.title(f"Words for Potential Trend Cluster {cluster_id} (Score: {row['TrendScore_Normalized']:.1f})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'adv_wordcloud_cluster_{cluster_id}.png')); plt.close()
        print(f"  - Saved word cloud for Cluster {cluster_id}")
    except Exception as e: print(f"  - WordCloud Error for Cluster {cluster_id}: {e}")

print(f"--- [trend_analysis_advanced.py] Advanced Trend Analysis Finished ---")

# import pandas as pd
# import numpy as np
# import re
# import nltk
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import Counter
# from wordcloud import WordCloud
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

# # Download necessary NLTK data
# nltk.download('stopwords')
# nltk.download('punkt')

# # Load YouTube comments dataset
# file_path = 'scrapers/csv_outputs/youtube_data.csv'  # Change path if needed
# df = pd.read_csv(file_path)

# # Ensure the 'Text' column exists
# if 'Text' not in df.columns:
#     raise ValueError("Dataset must have a 'Text' column containing the comments.")

# # Preprocessing Function
# stop_words = set(stopwords.words('english'))
# def preprocess_text(text):
#     text = re.sub(r'http\S+|[^\w\s]', ' ', str(text))  # Remove URLs and special characters
#     text = text.lower()
#     words = nltk.word_tokenize(text)
#     words = [word for word in words if word not in stop_words]  # Remove stopwords
#     return ' '.join(words)

# # Apply preprocessing
# df['cleaned_text'] = df['Text'].apply(preprocess_text)

# # Convert text into numerical vectors using TF-IDF
# vectorizer = TfidfVectorizer(max_features=1000)
# X = vectorizer.fit_transform(df['cleaned_text'])

# # 🔥 **1️⃣ Determine Optimal Number of Clusters Using Elbow Method**
# wcss = []
# silhouette_scores = []
# K_range = range(5, 15)  # Trying different cluster numbers

# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)  # Inertia = WCSS
#     silhouette_scores.append(silhouette_score(X, kmeans.labels_))  # Measure clustering quality

# # **Plot Elbow Curve**
# plt.figure(figsize=(8, 5))
# plt.plot(K_range, wcss, marker='o', linestyle='-', color='b')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
# plt.title('Elbow Method for Optimal Clusters')
# plt.show()

# # **Plot Silhouette Score Curve**
# plt.figure(figsize=(8, 5))
# plt.plot(K_range, silhouette_scores, marker='o', linestyle='-', color='g')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score vs Number of Clusters')
# plt.show()

# # ✅ **Choose the Best Number of Clusters Based on the Elbow Point**
# optimal_clusters = 13  # Change this after checking the elbow curve

# # 🔥 **2️⃣ Apply K-Means with Optimal Clusters**
# kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
# df['cluster'] = kmeans.fit_predict(X)

# # **Trending Topics Extraction**
# trending_topics = {}
# for cluster_num in range(optimal_clusters):
#     cluster_comments = df[df['cluster'] == cluster_num]['cleaned_text']
#     all_words = ' '.join(cluster_comments).split()
#     trending_topics[cluster_num] = Counter(all_words).most_common(10)

# # Print trending topics
# print("\n🔍 **Trending Topics (Top Words per Cluster):**")
# for cluster, words in trending_topics.items():
#     print(f"\nCluster {cluster}: {words}")

# # 📊 **3️⃣ Visualizing Trending Topics**
# plt.figure(figsize=(10, 5))
# sns.countplot(x=df['cluster'], palette='viridis')
# plt.xlabel('Topic Cluster')
# plt.ylabel('Number of Comments')
# plt.title('Trending Topics Distribution')
# plt.show()

# # 🌥️ **4️⃣ Generate Word Cloud for Most Popular Topic**
# most_popular_cluster = df['cluster'].value_counts().idxmax()
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(
#     ' '.join(df[df['cluster'] == most_popular_cluster]['cleaned_text'])
# )

# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.title(f"Trending Words in Cluster {most_popular_cluster}")
# plt.show()

# # 📁 Save analyzed data
# output_file = 'trend_analysis_youtube_data_ml.csv'
# df.to_csv(output_file, index=False)

# print("\nAnalysis Complete!")
