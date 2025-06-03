import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis
import pyLDAvis.gensim_models

def plot_top_hashtags(df, top_n=10):
    """
    Plot a horizontal bar chart for the top N hashtags.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns 'hashtag' and 'count'
        top_n (int): Number of top hashtags to display
    """
    top_tags = df.head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='count', y='hashtag', data=top_tags, palette='viridis')
    plt.title(f"Top {top_n} Bitcoin Hashtags")
    plt.xlabel("Count")
    plt.ylabel("Hashtag")
    plt.tight_layout()
    plt.savefig('plots/top_hashtags.png')
    plt.show()

def plot_top_users(df, top_n=10):
    """
    Plot a horizontal bar chart for the top N active users.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns 'user_name' and 'tweet_count'
        top_n (int): Number of top users to display
    """
    top_users = df.head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='tweet_count', y='user_name', data=top_users, palette='magma')
    plt.title(f"Top {top_n} Active Users")
    plt.xlabel("Tweet Count")
    plt.ylabel("User")
    plt.tight_layout()
    plt.savefig('plots/top_users.png')
    plt.show()

def visualize_topics(lda_model, corpus, dictionary):
    """
    Visualize LDA topics using pyLDAvis.
    
    Parameters:
        lda_model : Gensim LdaModel object
        corpus : Corpus used for LDA model
        dictionary : Gensim Dictionary object
    """
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.show(vis)
