import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)            # remove mentions
    text = re.sub(r"#", "", text)              # remove hashtag symbol
    text = re.sub(r"[^\w\s]", "", text)        # remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text
