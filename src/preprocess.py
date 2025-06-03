import re
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure nltk data is downloaded before running this script:
# nltk.download('stopwords')
# nltk.download('punkt')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean tweet text by:
    - Lowercasing
    - Removing URLs, mentions, hashtags, special characters, numbers
    - Removing stopwords
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove mentions and hashtags (we will process hashtags separately)
    text = re.sub(r"[@#]\w+", "", text)
    # Remove non-alphabetic characters and numbers
    text = re.sub(r"[^a-z\s]", "", text)
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def extract_hashtags(hashtags_series):
    """
    Extract hashtags from the 'hashtags' column of the DataFrame.
    The column contains string representations of lists, e.g. "['#bitcoin', '#btc']"
    Returns a DataFrame with hashtag counts sorted descending.
    """
    all_hashtags = []
    for entry in hashtags_series.dropna():
        # Remove brackets and quotes, split by comma
        cleaned = entry.strip("[]").replace("'", "").replace('"', "")
        tags = [tag.strip().lower() for tag in cleaned.split(",") if tag.strip()]
        all_hashtags.extend(tags)

    counts = Counter(all_hashtags)
    df_counts = pd.DataFrame(counts.items(), columns=['hashtag', 'count']).sort_values(by='count', ascending=False)
    return df_counts

def extract_active_users(df):
    """
    Analyze top active users by counting occurrences in 'user_name' column.
    Returns a DataFrame with username and tweet count.
    """
    if 'user_name' not in df.columns:
        raise ValueError("DataFrame must have a 'user_name' column.")
    counts = df['user_name'].value_counts().reset_index()
    counts.columns = ['user_name', 'tweet_count']
    return counts

def preprocess_data(df):
    """
    Apply cleaning to the 'text' column of the dataframe.
    Returns a new DataFrame with a cleaned text column added.
    """
    if 'text' not in df.columns:
        raise ValueError("DataFrame must have a 'text' column.")
    df = df.copy()
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df
