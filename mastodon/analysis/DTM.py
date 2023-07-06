import gensim
# from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import LdaSeqModel
from gensim.models.callbacks import CallbackAny2Vec

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import pandas as pd
import numpy as np
import logging
import datetime
import os
import statistics

from mastodon.settings import config

DATA_FILE = config.DATA_FOLDER + 'data.csv'

def do() -> None:
    """
    Performing topic modeling on time-series data involves analyzing how the topics of discussion change over time.
    To do this, you we use a variation of LDA (Latent Dirichlet Allocation) known as Dynamic Topic Modeling.
    Dynamic Topic Modeling extends LDA by allowing topics to change over time, which is useful for analyzing time-series
    data such as Mastodon toots. This approach models the topic distribution at each time point as a combination of the
    previous time point's topic distribution and a new set of topics.
    """

    # Load the csv file in a Pandas dataframe.
    df = pd.read_csv(DATA_FILE)

    # Preprocess the text data, and tokenize it.
    # tokenized_data = [simple_preprocess(text) for text in df['cleaned_content']]
    # Otherwise...
    # Assuming the data have been already preprocessed.
    tokenized_data = [text.split() for text in df['cleaned_content']]

    # gensim.corpora.dictionary.Dictionary: every unique token is associated with an unique incremental integer value.
    dictionary = Dictionary(tokenized_data)
    """
    BoW stands for Bag of Words. It is a commonly used technique in natural language processing (NLP) where a text 
    document is represented as a bag of its words, disregarding grammar and word order but keeping track of the 
    frequency of each word. This approach converts a text document into a numerical vector, which can be used as 
    input to machine learning algorithms for tasks such as text classification, clustering, and topic modeling. 
    The BoW representation can be obtained by first creating a dictionary of all unique words in the corpus and 
    then counting the occurrences of each word in each document. The resulting vector is often sparse, meaning that m
    ost of the entries are zero, and it is typically normalized to have unit length.
    
    The doc2bow method converts a document represented as a list of words into its BoW representation using the 
    provided dictionary.
    
    Just an example converted document: 
        [
            ...,
            [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), 
            (13, 1), (14, 1), (15, 1)],
            ...
        ]    
    """
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_data]

    """
        Check the data.
    """
    empty_entries = [i for i, doc in enumerate(corpus) if not doc]
    if empty_entries:
        raise ValueError(f'The corpus contains {len(empty_entries)} empty entries: {empty_entries}')
    else:
        # The corpus does not contain empty entries.
        pass

    """
        Create timestamped folders. 
    """
    # Get the current timestamp and format it.
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    # Create a subfolder with the current timestamp.
    models_subfolder_path = os.path.join(config.DTM_MODELS_FOLDER, timestamp)
    os.makedirs(models_subfolder_path, exist_ok=True)
    #
    logs_subfolder_path = os.path.join(config.LOGS_FOLDER, timestamp)
    os.makedirs(logs_subfolder_path, exist_ok=True)

    """
        Save the dictionary and the corpus to disk.
    """
    corpus_filepath = os.path.join(models_subfolder_path, 'corpus.mm')
    dictionary_filepath = os.path.join(models_subfolder_path, 'dictionary.pkl')
    #
    gensim.corpora.MmCorpus.serialize(corpus_filepath, corpus)
    dictionary.save(dictionary_filepath)

    """
        Time slicing of the toots.
    """
    # This will create a list of time slices where each time slice covers one day.
    # time_slice = df.groupby('date').size().tolist()
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['date_3days'] = df['date'].apply(lambda x: x - pd.Timedelta(x.day % 3, unit='D'))
    # Group the data by the new 'date_3days' column and count the number of items in each group
    time_slice = df.groupby('date_3days').size().tolist()

    """
        Logging.
    """
    # Set up logging to output to both the console and a file.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Set up console handler.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s : %(levelname)s : %(message)s'))
    logger.addHandler(console_handler)

    # Set up file handler.
    logs_filepath = os.path.join(logs_subfolder_path, 'model_output.log')
    file_handler = logging.FileHandler(logs_filepath)
    file_handler.setFormatter(logging.Formatter('%(asctime)s : %(levelname)s : %(message)s'))
    logger.addHandler(file_handler)

    """
        Training.
    """
    """
    Note that the num_topics parameter specifies the number of topics to be identified, and the time_slice parameter 
    can be used to specify how the time slices should be divided (e.g., by day, week, or month). 
    The passes parameter controls the number of times the model should iterate over the corpus during training.
    """
    num_topics = 5
    lda_seq = LdaSeqModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=20, time_slice=time_slice)
    # Save the model.
    model_filepath = os.path.join(models_subfolder_path, 'lda_seq_model.gensim')
    lda_seq.save(model_filepath)

    """
    Results.
    
    # Visualize the final results: the topics.
    vis = pyLDAvis.gensim_models.prepare(lda_seq, corpus, dictionary)
    results_filepath = os.path.join(models_subfolder_path, 'model_output.html')
    pyLDAvis.save_html(vis, results_filepath)
    # pyLDAvis.display(vis)
    """