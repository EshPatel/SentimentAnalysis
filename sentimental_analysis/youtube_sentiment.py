import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load CSV data
file_path = 'scrapers/csv_outputs/youtube_data.csv'  # Adjust path as needed
df = pd.read_csv(file_path)

# Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'http\S+|[^\w\s]', ' ', str(text))  # Remove links & special chars
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['cleaned_text'] = df['Text'].apply(preprocess_text)

# Combine VADER and TextBlob sentiment
sid = SentimentIntensityAnalyzer()

def combined_sentiment(text):
    blob_score = TextBlob(text).sentiment.polarity
    vader_score = sid.polarity_scores(text)['compound']
    return (blob_score + vader_score) / 2

df['sentiment_score'] = df['cleaned_text'].apply(combined_sentiment)

# Categorize sentiment with adjusted threshold
def categorize_sentiment(score):
    if score > 0.01:
        return 'Positive'
    elif score < -0.01:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)

# Histogram of sentiment scores
plt.figure(figsize=(10, 6))
df['sentiment_score'].hist(bins=10, range=(-1, 1), color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Sentiment Score Distribution (Combined VADER + TextBlob)')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Bar chart of sentiment categories
plt.figure(figsize=(8, 6))
categories_order = ['Positive', 'Neutral', 'Negative']
custom_palette = {
    'Positive': '#4CAF50',   # green
    'Neutral': '#FFEB3B',    # yellow
    'Negative': '#F44336'    # red
}

ax = sns.countplot(x='sentiment_category', data=df, order=categories_order, palette=custom_palette)
total = len(df)
for p in ax.patches:
    percentage = f"{100 * p.get_height() / total:.1f}%"
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.title('Sentiment Category Distribution')
plt.xlabel('Sentiment Category')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Word Cloud for each sentiment
for sentiment in ['Positive', 'Neutral', 'Negative']:
    text = ' '.join(df[df['sentiment_category'] == sentiment]['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Most Common Words in {sentiment} Comments")
    plt.show()

# Save analyzed data
output_file = 'scrapers/csv_outputs/analyzed_youtube_data.csv'
df.to_csv(output_file, index=False)

# Summary
print("Sentiment Category Counts:")
print(df['sentiment_category'].value_counts())

print("\nSample of Cleaned and Analyzed Data:")
print(df[['cleaned_text', 'sentiment_score', 'sentiment_category']].head())


# import pandas as pd
# import re
# import nltk
# import matplotlib.pyplot as plt
# import seaborn as sns
# from textblob import TextBlob
# from wordcloud import WordCloud
# from nltk.corpus import stopwords
# from nltk.sentiment import SentimentIntensityAnalyzer

# # Load the CSV data
# file_path = 'scrapers/csv_outputs/youtube_data.csv'  # Change this path if needed
# df = pd.read_csv(file_path)

# # Check if 'Text' column exists
# if 'Text' not in df.columns:
#     raise ValueError("Dataset must have a 'Text' column containing the comments.")

# # Preprocess text data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

# stop_words = set(stopwords.words('english'))

# def preprocess_text(text):
#     """Clean and preprocess text."""
#     text = re.sub(r'http\S+|[^\w\s]', ' ', str(text))  # Remove URLs and special characters
#     text = text.lower()  # Convert to lowercase
#     words = nltk.word_tokenize(text)  # Tokenize text
#     words = [word for word in words if word not in stop_words]  # Remove stopwords
#     return ' '.join(words)

# # Apply preprocessing
# df['cleaned_text'] = df['Text'].apply(preprocess_text)

# # Perform sentiment analysis using SentimentIntensityAnalyzer
# sid = SentimentIntensityAnalyzer()
# df['sentiment_score'] = df['cleaned_text'].apply(lambda text: sid.polarity_scores(text)['compound'])

# # Categorize Sentiment
# def categorize_sentiment(score):
#     if score > 0.2:
#         return 'Positive'
#     elif score < -0.2:
#         return 'Negative'
#     else:
#         return 'Neutral'

# df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)

# # Set up figure for all visualizations
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 2 rows, 2 columns

# # 🎯 1️⃣ Sentiment Score Histogram
# axes[0, 0].hist(df['sentiment_score'], bins=10, range=(-1, 1), color='skyblue', edgecolor='black', alpha=0.7)
# axes[0, 0].set_title('Sentiment Analysis of YouTube Comments')
# axes[0, 0].set_xlabel('Sentiment Score')
# axes[0, 0].set_ylabel('Frequency')
# axes[0, 0].grid(True)

# # 📊 2️⃣ Sentiment Category Distribution Bar Chart
# categories_order = ['Positive', 'Neutral', 'Negative']
# sns.countplot(x='sentiment_category', data=df, palette='viridis', order=categories_order, ax=axes[0, 1])
# axes[0, 1].set_title('Sentiment Category Distribution')
# axes[0, 1].set_xlabel('Sentiment Category')
# axes[0, 1].set_ylabel('Count')

# # Add percentage labels on the bar chart
# total = len(df)
# for p in axes[0, 1].patches:
#     percentage = f"{100 * p.get_height() / total:.1f}%"  # Format percentage
#     x = p.get_x() + p.get_width() / 2
#     y = p.get_height()
#     axes[0, 1].annotate(percentage, (x, y), ha='center', va='bottom')

# # 🌥️ 3️⃣ Word Cloud for Most Common Words (Across All Comments)
# text = ' '.join(df['cleaned_text'])
# wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)

# axes[1, 0].imshow(wordcloud, interpolation='bilinear')
# axes[1, 0].axis('off')
# axes[1, 0].set_title('Most Common Words in YouTube Comments')

# # 🌟 4️⃣ Word Cloud for Positive Comments
# positive_text = ' '.join(df[df['sentiment_category'] == 'Positive']['cleaned_text'])
# positive_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(positive_text)

# axes[1, 1].imshow(positive_wordcloud, interpolation='bilinear')
# axes[1, 1].axis('off')
# axes[1, 1].set_title('Most Common Words in Positive Comments')

# # Adjust layout for better readability
# plt.tight_layout()
# plt.show()

# # Save the analyzed data to a new CSV file
# output_file = 'analyzed_youtube_data.csv'
# df.to_csv(output_file, index=False)

# # Print sentiment category counts
# print("\n🔍 Sentiment Category Counts:")
# print(df['sentiment_category'].value_counts())

# # Print a sample of the cleaned and analyzed data
# print("\n📊 Sample of Cleaned and Analyzed Data:")
# print(df[['cleaned_text', 'sentiment_score', 'sentiment_category']].head())
