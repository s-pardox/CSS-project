import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from typing import Tuple

from mastodon.settings import config
from mastodon.utils import data_mng
def do(filepath: str, num_topics: int, start_date: str = None, end_date: str = None) -> Tuple[list, pd.DataFrame]:
    """
    This function applies Latent Dirichlet Allocation (LDA) to a dataset to discover the underlying topics.
    The input data is first tokenized, and then transformed into a document-term matrix. An LDA model is then
    trained on this matrix to uncover the underlying topics. The top 10 words for each topic are printed.
    Optionally, the data can be filtered to include only records between a specified start and end date.

    :param filepath: str, The path to the CSV file containing the data.
    :param num_topics: int, The number of topics for the LDA model to uncover.
    :param start_date: str, optional, Start date for time period selection in 'YYYY-MM-DD' format (default: None).
    :param end_date: str, optional, End date for time period selection in 'YYYY-MM-DD' format (default: None).
    :return: Tuple[list, pd.DataFrame], Returns a list of topics where each topic is represented as a list of words,
             and a DataFrame used in the LDA analysis.
    """
    df = pd.read_csv(filepath)

    # Convert date column to datetime if not already.
    df['date'] = pd.to_datetime(df['date'])

    # Filter data based on date range if provided.
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    print(f"\nAnalyzing data from {start_date} to {end_date}...")

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    df['tokenized_content'] = df['content'].apply(lambda x: tokenizer.tokenize(x.lower()))

    # Use ngram_range=(1, 1) to focus only on unigrams.
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1))
    doc_term_matrix = vectorizer.fit_transform(df['tokenized_content'].apply(lambda x: ' '.join(x)))

    lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online',
                                          random_state=100, verbose=1)
    lda_model.fit(doc_term_matrix)

    feature_names = list(vectorizer.vocabulary_.keys())

    topics = []
    for i, topic in enumerate(lda_model.components_):
        topic_keywords = [feature_names[index] for index in topic.argsort()[-10:]]
        topics.append(topic_keywords)
        print(f"\nTopic {i}:")
        print(', '.join(topic_keywords))

    return topics, df
