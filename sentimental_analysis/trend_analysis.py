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

DEFAULT_TOP_N_CLUSTERS_FOR_DEEP_DIVE = 3
DEFAULT_OPTIMAL_CLUSTERS = 5 # Default, user should determine from plots

nltk_resources = {'stopwords': 'stopwords', 'punkt': 'punkt'}
for resource, package in nltk_resources.items():
    try: nltk.data.find(f'corpora/{resource}')
    except: nltk.download(package, quiet=True)
stop_words_list = stopwords.words('english')


def perform_trend_analysis(input_analyzed_csv_path, plot_output_dir, timestamp_col_name=None, optimal_clusters_override=None):
    print(f"\n--- [trend_analysis.py] Callable Trend Analysis Started ---")
    print(f"[trend_analysis.py] Input sentiment-analyzed CSV: {input_analyzed_csv_path}")
    print(f"[trend_analysis.py] Plot output directory: {plot_output_dir}")
    if timestamp_col_name:
        print(f"[trend_analysis.py] Timestamp column: {timestamp_col_name}")
    else:
        print(f"[trend_analysis.py] No timestamp column provided for temporal analysis.")

    # Ensure plot output directory exists
    os.makedirs(plot_output_dir, exist_ok=True)

    try:
        df = pd.read_csv(input_analyzed_csv_path)
        print(f"[trend_analysis.py] Loaded {len(df)} rows from {input_analyzed_csv_path}")
    except FileNotFoundError:
        error_msg = f"Trend analysis input CSV not found: {input_analyzed_csv_path}"
        print(f"[trend_analysis.py] ERROR: {error_msg}")
        return {"error": error_msg, "plots": {}, "detailed_predictions_csv_path": None, "cluster_summary_csv_path": None}

    # --- Verify Required Columns from sentiment analysis (using 'cleaned_text' as per your file) ---
    required_input_cols = ['cleaned_text', 'sentiment_score', 'sentiment_category']
    if timestamp_col_name:
        required_input_cols.append(timestamp_col_name)

    missing_input_cols = [col for col in required_input_cols if col not in df.columns and col != timestamp_col_name]
    if missing_input_cols:
        error_msg = f"Trend analysis input CSV missing required columns: {missing_input_cols}. Expects 'cleaned_text'."
        print(f"[trend_analysis.py] ERROR: {error_msg}")
        return {"error": error_msg, "plots": {}, "detailed_predictions_csv_path": None, "cluster_summary_csv_path": None}

    has_timestamp = timestamp_col_name and timestamp_col_name in df.columns
    if timestamp_col_name and not has_timestamp:
        print(f"[trend_analysis.py] WARNING: Specified timestamp column '{timestamp_col_name}' not found. Temporal analysis limited.")

    df.dropna(subset=['cleaned_text'], inplace=True)
    if df.empty:
        error_msg = "No valid text data after dropping NaNs for trend analysis."
        print(f"[trend_analysis.py] ERROR: {error_msg}")
        return {"error": error_msg, "plots": {}, "detailed_predictions_csv_path": None, "cluster_summary_csv_path": None}
    print(f"[trend_analysis.py] Data ready for trend analysis: {len(df)} rows.")

    print("[trend_analysis.py] Vectorizing 'cleaned_text' (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=2000, stop_words=stop_words_list, ngram_range=(1,2)) # As per your file
    X = vectorizer.fit_transform(df['cleaned_text']) # Using 'cleaned_text'
    print(f"[trend_analysis.py] TF-IDF matrix shape: {X.shape}")
    feature_names = vectorizer.get_feature_names_out()

    # --- Output dictionary for paths ---
    plot_paths_dict = {} # To store paths of generated plots

    print("[trend_analysis.py] Determining optimal number of clusters (5-15)...")
    wcss = []
    silhouette_scores_list = [] # Renamed to avoid conflict
    K_range = range(5, 16)
    for k_val in K_range:
        km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        km.fit(X)
        wcss.append(km.inertia_)
        if X.shape[0] > 1 and X.shape[0] <= 5000: # Avoid for very large datasets
            try: silhouette_scores_list.append(silhouette_score(X, km.labels_))
            except: silhouette_scores_list.append(np.nan)
        else: silhouette_scores_list.append(np.nan)

    # Elbow Plot
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, wcss, marker='o', color='b')
    plt.title('Elbow Method for Optimal k'); plt.xlabel('Number of Clusters (k)'); plt.ylabel('WCSS (Inertia)')
    plt.xticks(K_range); plt.grid(True)
    elbow_plot_filename = 'trend_elbow_plot.png' # Simpler filename
    elbow_plot_path = os.path.join(plot_output_dir, elbow_plot_filename)
    plt.savefig(elbow_plot_path); plt.close()
    plot_paths_dict['elbow_plot'] = elbow_plot_path
    print(f"[trend_analysis.py] Saved plot: {elbow_plot_filename}")

    # Silhouette Plot
    if any(not np.isnan(s) for s in silhouette_scores_list):
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, silhouette_scores_list, marker='o', color='g')
        plt.title('Silhouette Score vs. Number of Clusters'); plt.xlabel('Number of Clusters (k)'); plt.ylabel('Silhouette Score')
        plt.xticks(K_range); plt.grid(True)
        silhouette_plot_filename = 'trend_silhouette_plot.png'
        silhouette_plot_path = os.path.join(plot_output_dir, silhouette_plot_filename)
        plt.savefig(silhouette_plot_path); plt.close()
        plot_paths_dict['silhouette_plot'] = silhouette_plot_path
        print(f"[trend_analysis.py] Saved plot: {silhouette_plot_filename}")
        try:
            valid_scores = [s for s in silhouette_scores_list if not np.isnan(s)]
            if valid_scores:
                # Find k with max score if silhouette scores were computed
                optimal_k_suggestion = K_range.start + np.nanargmax(silhouette_scores_list)
                print(f"[trend_analysis.py] Suggestion from Silhouette: Optimal k ~ {optimal_k_suggestion}")
        except: pass

    # Use override if provided, else default
    optimal_k_to_use = optimal_clusters_override if optimal_clusters_override is not None else DEFAULT_OPTIMAL_CLUSTERS
    # This line `optimal_clusters = 5` from your original file is now effectively replaced by optimal_k_to_use
    print(f"[trend_analysis.py] Using k = {optimal_k_to_use} clusters.")

    kmeans = KMeans(n_clusters=optimal_k_to_use, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    order_centroids = centroids.argsort()[:, ::-1]

    print("[trend_analysis.py] Analyzing cluster characteristics...")
    cluster_analysis = []
    terms_per_cluster_display = 10
    for i in range(optimal_k_to_use): # Use the determined k
        cluster_df = df[df['cluster'] == i]
        if cluster_df.empty: continue
        top_terms = [feature_names[ind] for ind in order_centroids[i, :terms_per_cluster_display]]
        cluster_info = {
            'Cluster': i, 'Size': len(cluster_df),
            'AvgSentiment': cluster_df['sentiment_score'].mean(),
            'PositiveRatio': cluster_df[cluster_df['sentiment_category'] == 'Positive'].shape[0] / len(cluster_df) if len(cluster_df) > 0 else 0,
            'TopTerms': ', '.join(top_terms) # This will use 'cleaned_text' based terms
        }
        cluster_analysis.append(cluster_info)
    
    if not cluster_analysis: # Handle case where no clusters were populated (e.g., k=0 or empty df)
        error_msg = "No clusters were analyzed. Data might be too sparse or k too small."
        print(f"[trend_analysis.py] WARNING: {error_msg}")
        return {"error": error_msg, "plots": plot_paths_dict, "detailed_predictions_csv_path": None, "cluster_summary_csv_path": None}

    cluster_summary_df = pd.DataFrame(cluster_analysis)
    # Check if DataFrame is empty before calculating TrendScore
    if cluster_summary_df.empty:
        error_msg = "Cluster summary DataFrame is empty. Cannot calculate TrendScore."
        print(f"[trend_analysis.py] WARNING: {error_msg}")
        # Proceed with saving what we have, but paths for CSVs might be None
        detailed_csv_output_path = os.path.join(os.path.dirname(str(input_analyzed_csv_path)), 'advanced_trend_analysis_predictions.csv')
        df.to_csv(detailed_csv_output_path, index=False) # Save the df with 'cluster' column
        return {"error": None, "plots": plot_paths_dict, "detailed_predictions_csv_path": detailed_csv_output_path, "cluster_summary_csv_path": None}


    cluster_summary_df['TrendScore'] = (np.log1p(cluster_summary_df['Size']) *
                                        (cluster_summary_df['AvgSentiment'] + 1.1) *
                                        (cluster_summary_df['PositiveRatio'] + 0.1))
    min_score, max_score = cluster_summary_df['TrendScore'].min(), cluster_summary_df['TrendScore'].max()
    cluster_summary_df['TrendScore_Normalized'] = 100 * (cluster_summary_df['TrendScore'] - min_score) / (max_score - min_score) if max_score > min_score else 50.0
    cluster_summary_df = cluster_summary_df.sort_values(by='TrendScore_Normalized', ascending=False).reset_index(drop=True)

    # --- Temporal Analysis (if has_timestamp) ---
    if has_timestamp:
        print(f"[trend_analysis.py] Performing temporal analysis using: {timestamp_col_name}...")
        df_temp = df.copy() # Work on a copy for temporal modifications
        df_temp[timestamp_col_name] = pd.to_datetime(df_temp[timestamp_col_name], errors='coerce')
        df_temp.dropna(subset=[timestamp_col_name], inplace=True)
        if not df_temp.empty:
            df_sorted_time = df_temp.sort_values(by=timestamp_col_name)
            df_sorted_time['TimePeriod'] = df_sorted_time[timestamp_col_name].dt.to_period('W')
            
            temporal_cluster_summary_agg = df_sorted_time.groupby(['cluster', 'TimePeriod']).agg(
                PeriodSize=('sentiment_score', 'size'),
                PeriodAvgSentiment=('sentiment_score', 'mean')
            ).reset_index()

            for idx, row in cluster_summary_df.head(DEFAULT_TOP_N_CLUSTERS_FOR_DEEP_DIVE).iterrows():
                cluster_id = row['Cluster']
                data_c = temporal_cluster_summary_agg[temporal_cluster_summary_agg['cluster'] == cluster_id]
                if len(data_c) > 1: # Need at least two periods to show a trend
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
                    temp_plot_filename = f'trend_temporal_cluster_{cluster_id}.png'
                    temp_plot_path = os.path.join(plot_output_dir, temp_plot_filename)
                    plt.savefig(temp_plot_path); plt.close()
                    plot_paths_dict[f'temporal_trend_cluster_{cluster_id}'] = temp_plot_path
                    print(f"[trend_analysis.py] Saved temporal plot: {temp_plot_filename}")
            # Simplified Burst Detection (using 'cleaned_text')
            # ... (Keep burst detection logic, ensuring it uses 'cleaned_text' from df_sorted_time)
            # ... and ensure any generated plot paths are added to plot_paths_dict
        else:
            print(f"[trend_analysis.py] Not enough valid timestamp data for temporal plots.")


    # Network Analysis (using 'cleaned_text')
    # ... (Keep network analysis logic, ensuring it uses 'cleaned_text')
    # ... and ensure any generated plot paths are added to plot_paths_dict
    if not cluster_summary_df.empty:
        top_cluster_for_network = cluster_summary_df.iloc[0]['Cluster']
        # ... (rest of your network code)
        network_plot_filename = f'trend_network_cluster_{top_cluster_for_network}.png'
        network_plot_path = os.path.join(plot_output_dir, network_plot_filename)
        # if plot is saved: plot_paths_dict[f'network_cluster_{top_cluster_for_network}'] = network_plot_path
        # For brevity, assuming the plot saving part of network analysis is there
        # Example: plt.savefig(network_plot_path); plt.close(); plot_paths_dict[...] = network_plot_path

    # --- Final Output CSVs and Plots from Cluster Summary ---
    print("\n" + "="*50)
    print(" Potential Trend Clusters (Ranked by Score) ")
    print("="*50)
    if not cluster_summary_df.empty:
        pd.options.display.float_format = '{:.3f}'.format
        pd.options.display.max_colwidth = 100 
        print(cluster_summary_df.to_string(index=False))
        pd.reset_option('display.float_format')
        pd.reset_option('display.max_colwidth')
    else:
        print("No cluster summary to display.")
    print("="*50 + "\n")

    # Detailed predictions (df with 'cluster' column)
    detailed_csv_output_filename = 'advanced_trend_analysis_predictions.csv' # As per your trend_analysis.py
    detailed_csv_output_path = os.path.join(os.path.dirname(str(input_analyzed_csv_path)), detailed_csv_output_filename)
    df.to_csv(detailed_csv_output_path, index=False) # Save the df with 'cluster' column
    print(f"[trend_analysis.py] Saved enriched DataFrame to: {detailed_csv_output_path}")

    # Cluster summary CSV
    cluster_summary_filename = 'adv_cluster_trend_summary.csv' # As per your trend_analysis.py
    cluster_summary_output_path = os.path.join(plot_output_dir, cluster_summary_filename) # Saved in plot dir
    if not cluster_summary_df.empty:
        cluster_summary_df.to_csv(cluster_summary_output_path, index=False, float_format="%.4f")
        print(f"[trend_analysis.py] Saved cluster summary to: {cluster_summary_output_path}")
    else:
        cluster_summary_output_path = None # No summary to save

    # Visualizations (Cluster Size, Sentiment Profile, Top N Word Clouds)
    if not cluster_summary_df.empty:
        # Trend Score Comparison Plot
        plt.figure(figsize=(12, 7))
        sns.barplot(x='Cluster', y='TrendScore_Normalized', data=cluster_summary_df.head(10),
                    palette='coolwarm_r', order=cluster_summary_df.head(10)['Cluster'])
        plt.title('Relative Trend Potential Score by Topic Cluster (Top 10)')
        plt.xlabel('Topic Cluster ID'); plt.ylabel('Normalized Trend Potential Score (0-100)')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        trend_score_plot_filename = 'trend_score_comparison.png'
        trend_score_plot_path = os.path.join(plot_output_dir, trend_score_plot_filename)
        plt.savefig(trend_score_plot_path); plt.close()
        plot_paths_dict['trend_score_comparison'] = trend_score_plot_path
        print(f"[trend_analysis.py] Saved plot: {trend_score_plot_filename}")

        # Sentiment Profile Plot
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x='Size', y='AvgSentiment', hue='Cluster', size='PositiveRatio',
            sizes=(50, 500), palette='viridis', data=cluster_summary_df, legend='brief')
        plt.title('Cluster Sentiment Profile'); plt.xlabel('Cluster Size'); plt.ylabel('Average Sentiment Score')
        plt.axhline(0, color='grey', linestyle='--', lw=0.8); plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(rect=[0, 0, 0.85, 1])
        sentiment_profile_plot_filename = 'trend_sentiment_profile.png'
        sentiment_profile_plot_path = os.path.join(plot_output_dir, sentiment_profile_plot_filename)
        plt.savefig(sentiment_profile_plot_path); plt.close()
        plot_paths_dict['sentiment_profile'] = sentiment_profile_plot_path
        print(f"[trend_analysis.py] Saved plot: {sentiment_profile_plot_filename}")

        # Word Clouds
        print("[trend_analysis.py] Generating word clouds for top trend clusters...")
        for i in range(min(DEFAULT_TOP_N_CLUSTERS_FOR_DEEP_DIVE, len(cluster_summary_df))):
            row = cluster_summary_df.iloc[i]
            cluster_id = row['Cluster']
            cluster_text = ' '.join(df[df['cluster'] == cluster_id]['cleaned_text'].astype(str)) # Using 'cleaned_text'
            if not cluster_text.strip(): continue
            try:
                wc = WordCloud(width=1000, height=500, background_color='white', colormap='viridis', max_words=75).generate(cluster_text)
                plt.figure(figsize=(10,5)); plt.imshow(wc, interpolation='bilinear'); plt.axis('off')
                plt.title(f"Words for Potential Trend Cluster {cluster_id} (Score: {row['TrendScore_Normalized']:.1f})")
                plt.tight_layout()
                wc_filename = f'trend_wordcloud_cluster_{cluster_id}.png'
                wc_path = os.path.join(plot_output_dir, wc_filename)
                plt.savefig(wc_path); plt.close()
                plot_paths_dict[f'wordcloud_cluster_{cluster_id}'] = wc_path
                print(f"  - Saved word cloud for Cluster {cluster_id}")
            except Exception as e: print(f"  - WordCloud Error for Cluster {cluster_id}: {e}")
    
    print(f"--- [trend_analysis.py] Callable Trend Analysis Finished ---")
    return {
        "error": None,
        "plots": plot_paths_dict,
        "detailed_predictions_csv_path": detailed_csv_output_path,
        "cluster_summary_csv_path": cluster_summary_output_path
    }

if __name__ == "__main__":
    # Standalone test logic for trend_analysis.py
    print("--- [trend_analysis.py] Standalone Test ---")
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming script is in project root for this standalone test, or in 'sentimental_analysis' subdir
    project_root_guess = current_script_dir 
    # if os.path.basename(current_script_dir) == "sentimental_analysis": # If in a subdir
    #    project_root_guess = os.path.dirname(current_script_dir)
        
    # This path should point to the output of youtube_sentiment.py
    test_input_csv_standalone = os.path.join(project_root_guess, 'scrapers', 'csv_outputs', 'analyzed_youtube_data.csv')
    test_standalone_plot_dir = os.path.join(project_root_guess, "presentation_outputs_trend_standalone_test")
    
    print(f"[trend_analysis.py] Standalone test: Input CSV path: {test_input_csv_standalone}")
    print(f"[trend_analysis.py] Standalone test: Plot output dir: {test_standalone_plot_dir}")

    if not os.path.exists(test_input_csv_standalone):
        print(f"[trend_analysis.py] Test input file '{test_input_csv_standalone}' not found.")
        print(f"[trend_analysis.py] Creating a dummy input file for testing trend_analysis.py standalone...")
        os.makedirs(os.path.dirname(test_input_csv_standalone), exist_ok=True)
        # Dummy data needs 'cleaned_text', 'sentiment_score', 'sentiment_category'
        dummy_data = {
            'Text': ['raw comment 1', 'raw comment 2', 'raw comment 3 about topic a', 'comment on topic b', 'topic a again'],
            'cleaned_text': ['comment one topic', 'second comment here', 'topic great stuff', 'topic b something else', 'topic fantastic'],
            'sentiment_score': [0.5, -0.2, 0.8, 0.1, 0.9],
            'sentiment_category': ['Positive', 'Negative', 'Positive', 'Neutral', 'Positive'],
            'PublishedAt': [datetime.now().isoformat()] * 5 # Dummy timestamp
        }
        pd.DataFrame(dummy_data).to_csv(test_input_csv_standalone, index=False)
        print(f"[trend_analysis.py] Dummy input file created at {test_input_csv_standalone}")

    results = perform_trend_analysis(
        input_analyzed_csv_path=test_input_csv_standalone,
        plot_output_dir=test_standalone_plot_dir,
        timestamp_col_name='PublishedAt', # Match dummy data
        optimal_clusters_override=3 # For quick test
    )
    if results["error"]:
        print(f"[trend_analysis.py] Error during standalone test: {results['error']}")
    else:
        print(f"[trend_analysis.py] Standalone test analysis complete. Results:")
        print(f"  Plots generated in: {test_standalone_plot_dir}")
        for name, path in results.get("plots", {}).items(): print(f"    - {name}: {path}")
        print(f"  Detailed predictions CSV: {results.get('detailed_predictions_csv_path')}")
        print(f"  Cluster summary CSV: {results.get('cluster_summary_csv_path')}")
    print("--- [trend_analysis.py] Standalone Test Finished ---")

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
