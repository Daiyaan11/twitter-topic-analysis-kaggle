import pandas as pd
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pyLDAvis.gensim_models
import pyLDAvis

def run_topic_modeling(cleaned_csv, num_topics=5, num_words=10):
    df = pd.read_csv(cleaned_csv)
    stop_words = set(stopwords.words('english'))

    # Tokenize + remove stopwords
    print("Tokenizing and removing stopwords...")
    texts = [
        [word for word in word_tokenize(doc.lower()) if word.isalpha() and word not in stop_words]
        for doc in df['clean_text']
    ]

    # Build dictionary + corpus
    print("Building dictionary and corpus...")
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # LDA model
    print(f"Running LDA with {num_topics} topics...")
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    # Print topics
    for idx, topic in lda_model.print_topics(num_topics=num_topics, num_words=num_words):
        print(f"Topic #{idx + 1}: {topic}")

    # Visualize with pyLDAvis
    print("Preparing LDA visualization...")
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'plots/lda_topics.html')
    print("LDA topics visualization saved as plots/lda_topics.html")
