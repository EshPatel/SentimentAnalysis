import pandas as pd
import re
import nltk
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import os

nltk_resources = {
    'tokenizers/punkt': 'punkt', 'corpora/stopwords': 'stopwords',
    'sentiment/vader_lexicon.zip': 'vader_lexicon', 'corpora/wordnet': 'wordnet',
    'corpora/omw-1.4': 'omw-1.4', 'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
    'punkt_tab': 'punkt_tab'
}
for path, resource_name in nltk_resources.items():
    try: nltk.data.find(path)
    except: nltk.download(resource_name, quiet=True)

stop_words_base = set(stopwords.words('english'))
generalized_custom_stopwords = {
    'video', 'channel', 'watch', 'watching', 'comment', 'comments', 'subscriber', 'subscribe', 'like', 'share', 'links', 'description',
    'pls', 'plz', 'thx', 'thank', 'thanks', 'u', 'ur', 'im', 'ive', 'id', 'ill', 'youre', 'hes', 'shes', 'theyre', 'isnt', 'arent', 'wasnt', 'werent',
    'lol', 'xd', 'rofl', 'lmao', 'omg', 'btw', 'tbh', 'imo', 'smh', 'fr', 'ngl', 'af', 'irl',
    'guy', 'guys', 'dude', 'bro', 'sis', 'girl', 'man', 'woman', 'sir', 'madam',
    'actually', 'literally', 'basically', 'really', 'just', 'even', 'also', 'well', 'still', 'though', 'although', 'however', 'maybe', 'perhaps',
    'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'lemme', 'aint',
    'thing', 'things', 'stuff', 'bit', 'lots', 'lot', 'one', 'way', 'times', 'day', 'days', 'week', 'month', 'year',
    'people', 'person', 'anyone', 'someone', 'everyone', 'everybody', 'nobody', 'another',
    'get', 'got', 'go', 'went', 'make', 'made', 'see', 'saw', 'know', 'knew', 'say', 'said', 'tell', 'told', 'ask', 'asked',
    'look', 'looks', 'feel', 'feels', 'seem', 'seems', 'try', 'uses', 'need', 'needs', 'want', 'wants', 'wish',
    'right', 'left', 'yes', 'yeah', 'yo', 'ok', 'okay', 'alright', 'true', 'false', 'sure', 'course', 'always', 'never',
    'number', 'numbers', 'fig', 'figure', 'image', 'photo', 'pic', 'http', 'https', 'www', 'com', 'org', 'html', 'php',
    'put', 'take', 'give', 'keep', 'let', 'become', 'include', 'continue', 'set', 'learn', 'understand', 'mean', 'means',
    'much', 'many', 'more', 'less', 'most', 'least', 'new', 'old', 'first', 'last', 'next', 'big', 'small', 'high', 'low',
    'nbsp', 'amp',
}
stop_words = stop_words_base.union(generalized_custom_stopwords)
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s\']', ' ', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = text.lower()
    words = nltk.word_tokenize(text)
    processed_words = []
    for w in words:
        if len(w) > 2 and "'" not in w:
            lemmatized_word = lemmatizer.lemmatize(w, get_wordnet_pos(w))
            if lemmatized_word not in stop_words:
                processed_words.append(lemmatized_word)
    return ' '.join(processed_words)

sid = SentimentIntensityAnalyzer()
def combined_sentiment(text):
    if not text.strip(): return 0.0
    blob_score = TextBlob(text).sentiment.polarity
    vader_score = sid.polarity_scores(text)['compound']
    return (blob_score + vader_score) / 2

def categorize_sentiment(score):
    if score > 0.05: return 'Positive'
    elif score < -0.05: return 'Negative'
    else: return 'Neutral'

def perform_sentiment_analysis_and_generate_plots(raw_scraped_csv_path, output_plot_dir_param):
    print(f"\n--- [youtube_sentiment.py] Sentiment Analysis Started ---")
    print(f"[youtube_sentiment.py] Reading raw scraped data from: {raw_scraped_csv_path}")
    print(f"[youtube_sentiment.py] Plots will be saved to: {output_plot_dir_param}")

    current_output_plot_dir = output_plot_dir_param
    os.makedirs(current_output_plot_dir, exist_ok=True)

    try:
        df = pd.read_csv(raw_scraped_csv_path)
        print(f"[youtube_sentiment.py] Successfully loaded {len(df)} rows from {raw_scraped_csv_path}")
        print(f"[youtube_sentiment.py] Columns in loaded raw CSV: {df.columns.tolist()}")
    except FileNotFoundError:
        error_msg = f"Input CSV not found: {raw_scraped_csv_path}"
        print(f"[youtube_sentiment.py] ERROR: {error_msg}")
        return {"error": error_msg, "plots": {}, "analyzed_csv_path": None}
    
    if 'Text' not in df.columns:
        if all(col in df.columns for col in ['cleaned_text', 'sentiment_score', 'sentiment_category']):
            print(f"[youtube_sentiment.py] WARNING: The input file {raw_scraped_csv_path} appears to be ALREADY ANALYZED.")
            print("[youtube_sentiment.py] This script expects raw scraped data with a 'Text' column. Re-analysis will occur.")
        else:
            error_msg = "Input CSV must contain a 'Text' column for raw data."
            print(f"[youtube_sentiment.py] ERROR: {error_msg}")
            return {"error": error_msg, "plots": {}, "analyzed_csv_path": None}

    source_text_column = 'Text'
    print(f"[youtube_sentiment.py] Preprocessing text from '{source_text_column}' column...")
    df['cleaned_text'] = df[source_text_column].apply(preprocess_text)
    print("[youtube_sentiment.py] Calculating sentiment scores...")
    df['sentiment_score'] = df['cleaned_text'].apply(combined_sentiment)
    df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)

    plot_paths = {}
    categories_order = ['Positive', 'Neutral', 'Negative']
    custom_palette = {'Positive': '#4CAF50', 'Neutral': '#FFEB3B', 'Negative': '#F44336'}

    print("[youtube_sentiment.py] Generating sentiment category distribution plot...")
    plt.figure(figsize=(8, 6))
    present_categories = [cat for cat in categories_order if cat in df['sentiment_category'].unique()]
    if present_categories:
        ax = sns.countplot(x='sentiment_category', data=df, order=present_categories, palette=custom_palette)
        total = len(df['sentiment_category'].dropna())
        for p in ax.patches:
            height = p.get_height()
            percentage_text = f"{100 * height / total:.1f}%" if total > 0 else "0.0%"
            ax.annotate(percentage_text, (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
        plt.title('Sentiment Category Distribution')
        plt.xlabel('Sentiment Category')
        plt.ylabel('Count')
        plt.tight_layout()
        bar_chart_filename = 'sentiment_category_distribution.png'
        bar_chart_path = os.path.join(current_output_plot_dir, bar_chart_filename)
        plt.savefig(bar_chart_path)
        plot_paths['category_distribution'] = bar_chart_path # Key by purpose
        plt.close()
        print(f"[youtube_sentiment.py] Saved plot: {bar_chart_path}")

    print("[youtube_sentiment.py] Generating word clouds...")
    for sentiment in categories_order:
        sentiment_texts_list = df[df['sentiment_category'] == sentiment]['cleaned_text'].astype(str).tolist()
        full_text_for_category = ' '.join(sentiment_texts_list)

        if not full_text_for_category.strip() and not sentiment_texts_list:
            print(f"[youtube_sentiment.py] No text for '{sentiment}' word cloud. Skipping.")
            continue

        wordcloud_title = f"Most Common Words in {sentiment} Comments"
        wordcloud_filename_stem = f'{sentiment.lower()}_wordcloud_rawfreq'
        use_frequencies_for_wc = False
        frequencies_data_for_wc = None

        if sentiment == 'Negative' and sentiment_texts_list:
            print(f"[youtube_sentiment.py] Applying TF-IDF for Negative comments...")
            try:
                vectorizer = TfidfVectorizer(max_features=200, stop_words=list(stop_words), ngram_range=(1, 2), min_df=2, max_df=0.85, use_idf=True, smooth_idf=True, sublinear_tf=True)
                tfidf_matrix = vectorizer.fit_transform(sentiment_texts_list)
                feature_names = vectorizer.get_feature_names_out()
                sum_tfidf = tfidf_matrix.sum(axis=0)
                tfidf_scores_dict = {feature_names[col]: sum_tfidf[0, col] for col in range(tfidf_matrix.shape[1])}
                if tfidf_scores_dict:
                    frequencies_data_for_wc = tfidf_scores_dict
                    use_frequencies_for_wc = True
                    wordcloud_title = f"Top TF-IDF Weighted Words in {sentiment} Comments"
                    wordcloud_filename_stem = f'{sentiment.lower()}_wordcloud_tfidf'
            except ValueError as e:
                print(f"[youtube_sentiment.py] TF-IDF Error ({sentiment}): {e}. Using raw if available.")
                if not full_text_for_category.strip(): continue

        if not use_frequencies_for_wc and not full_text_for_category.strip():
            continue
            
        try:
            wc = WordCloud(width=1200, height=600, background_color='white', stopwords=None, max_words=100, min_word_length=3, collocations=not use_frequencies_for_wc, colormap='viridis', random_state=42)
            if use_frequencies_for_wc and frequencies_data_for_wc:
                final_wordcloud = wc.generate_from_frequencies(frequencies_data_for_wc)
            elif full_text_for_category.strip():
                final_wordcloud = wc.generate(full_text_for_category)
            else:
                continue
            
            wc_filename = f'{wordcloud_filename_stem}.png'
            wc_path = os.path.join(current_output_plot_dir, wc_filename)
            plt.figure(figsize=(12, 6))
            plt.imshow(final_wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(wordcloud_title)
            plt.tight_layout(pad=0)
            plt.savefig(wc_path)
            plot_paths[wordcloud_filename_stem] = wc_path # Key by filename_stem
            plt.close()
            print(f"[youtube_sentiment.py] Saved word cloud: {wc_path}")
        except Exception as e:
            print(f"[youtube_sentiment.py] WordCloud Error ({sentiment}): {e}")

    analyzed_csv_filename = 'analyzed_youtube_data.csv'
    output_csv_base_dir = os.path.dirname(str(raw_scraped_csv_path))
    analyzed_csv_output_path = os.path.join(output_csv_base_dir, analyzed_csv_filename)
    df.to_csv(analyzed_csv_output_path, index=False)

    print(f"[youtube_sentiment.py] Analyzed data (with sentiment columns) saved to: {analyzed_csv_output_path}")
    print(f"[youtube_sentiment.py] Final DataFrame columns: {df.columns.tolist()}")
    print(f"[youtube_sentiment.py] Sentiment Category Value Counts:\n{df['sentiment_category'].value_counts(dropna=False)}")
    print(f"--- [youtube_sentiment.py] Sentiment Analysis Finished ---")

    return {"error": None, "plots": plot_paths, "analyzed_csv_path": analyzed_csv_output_path}

if __name__ == "__main__":
    print("--- [youtube_sentiment.py] Standalone Test ---")
    # Get the project root directory (assuming this script is in sentimental_analysis subdir)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir) 
    
    test_input_csv = os.path.join(project_root, 'scrapers', 'csv_outputs', 'youtube_data.csv')
    standalone_plot_dir = os.path.join(project_root, "presentation_outputs_standalone_test")
    
    print(f"[youtube_sentiment.py] Standalone test: Input CSV path: {test_input_csv}")
    print(f"[youtube_sentiment.py] Standalone test: Plot output dir: {standalone_plot_dir}")

    if not os.path.exists(test_input_csv):
        print(f"[youtube_sentiment.py] Test input file '{test_input_csv}' not found for standalone test.")
        print("[youtube_sentiment.py] Please run the scraper via app.py first to generate this file, or create a dummy file manually for testing.")
    else:
        print(f"[youtube_sentiment.py] Found '{test_input_csv}'. Proceeding with standalone test analysis.")
        results = perform_sentiment_analysis_and_generate_plots(
            raw_scraped_csv_path=test_input_csv,
            output_plot_dir_param=standalone_plot_dir
        )
        if results["error"]:
            print(f"[youtube_sentiment.py] Error during standalone test: {results['error']}")
        else:
            print(f"[youtube_sentiment.py] Standalone test analysis complete. Plots in '{standalone_plot_dir}':")
            if results["plots"]:
                for name, path in results["plots"].items(): print(f"- {name}: {path}")
            else:
                print("- No plots were generated in the test.")
            print(f"[youtube_sentiment.py] Analyzed CSV from standalone test: {results['analyzed_csv_path']}")
    print("--- [youtube_sentiment.py] Standalone Test Finished ---")



# import pandas as pd
# import re
# import nltk
# import matplotlib.pyplot as plt
# import seaborn as sns
# from textblob import TextBlob
# from wordcloud import WordCloud
# from nltk.corpus import stopwords
# from nltk.sentiment import SentimentIntensityAnalyzer

# # Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

# # Load CSV data
# file_path = 'scrapers/csv_outputs/youtube_data.csv'  # Adjust path as needed
# df = pd.read_csv(file_path)

# # Preprocessing
# stop_words = set(stopwords.words('english'))

# def preprocess_text(text):
#     text = re.sub(r'http\S+|[^\w\s]', ' ', str(text))  # Remove links & special chars
#     text = text.lower()
#     words = nltk.word_tokenize(text)
#     words = [word for word in words if word not in stop_words]
#     return ' '.join(words)

# df['cleaned_text'] = df['Text'].apply(preprocess_text)

# # Combine VADER and TextBlob sentiment
# sid = SentimentIntensityAnalyzer()

# def combined_sentiment(text):
#     blob_score = TextBlob(text).sentiment.polarity
#     vader_score = sid.polarity_scores(text)['compound']
#     return (blob_score + vader_score) / 2

# df['sentiment_score'] = df['cleaned_text'].apply(combined_sentiment)

# # Categorize sentiment with adjusted threshold
# def categorize_sentiment(score):
#     if score > 0.01:
#         return 'Positive'
#     elif score < -0.01:
#         return 'Negative'
#     else:
#         return 'Neutral'

# df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)

# # Histogram of sentiment scores
# plt.figure(figsize=(10, 6))
# df['sentiment_score'].hist(bins=10, range=(-1, 1), color='skyblue', edgecolor='black', alpha=0.7)
# plt.title('Sentiment Score Distribution (Combined VADER + TextBlob)')
# plt.xlabel('Sentiment Score')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

# # Bar chart of sentiment categories
# plt.figure(figsize=(8, 6))
# categories_order = ['Positive', 'Neutral', 'Negative']
# custom_palette = {
#     'Positive': '#4CAF50',
#     'Neutral': '#FFEB3B',
#     'Negative': '#F44336'
# }

# ax = sns.countplot(x='sentiment_category', data=df, order=categories_order, palette=custom_palette)
# total = len(df)
# for p in ax.patches:
#     percentage = f"{100 * p.get_height() / total:.1f}%"
#     x = p.get_x() + p.get_width() / 2
#     y = p.get_height()
#     ax.annotate(percentage, (x, y), ha='center', va='bottom')

# plt.title('Sentiment Category Distribution')
# plt.xlabel('Sentiment Category')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.show()

# # Word Cloud for each sentiment
# for sentiment in ['Positive', 'Neutral', 'Negative']:
#     text = ' '.join(df[df['sentiment_category'] == sentiment]['cleaned_text'])
#     wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.title(f"Most Common Words in {sentiment} Comments")
#     plt.show()

# # Save analyzed data
# output_file = 'scrapers/csv_outputs/analyzed_youtube_data.csv'
# df.to_csv(output_file, index=False)

# # Summary
# print("Sentiment Category Counts:")
# print(df['sentiment_category'].value_counts())

# print("\nSample of Cleaned and Analyzed Data:")
# print(df[['cleaned_text', 'sentiment_score', 'sentiment_category']].head())