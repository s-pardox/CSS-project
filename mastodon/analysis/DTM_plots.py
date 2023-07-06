import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import LdaSeqModel
from gensim.corpora import Dictionary, MmCorpus

from mastodon.settings import config
from mastodon.utils import data_mng

DATA_FILE = config.DATA_FOLDER + 'data.csv'

def do() -> None:
    pd.set_option('display.max_rows', 5000)
    pd.set_option('display.max_columns', 5000)
    pd.set_option('display.width', 10000)
    pd.set_option('display.max_colwidth', 1000)

    df = data_mng.load_from_csv_to_df(DATA_FILE)
    plot_time_slices_topics_chart(df)

def plot_time_slices_topics_chart(df: pd.DataFrame) -> None:
    """
    Plots a bar chart of the number of topics per time slice, with custom x-axis labels showing
    both the progressive number and the explicit date range. The x-axis labels are rotated
    90 degrees for proper display.

    :param df: DataFrame containing the data with 'date_3days' and 'date' columns
    :return: None
    """

    # This will create a list of time slices where each time slice covers 3 days.
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['date_3days'] = df['date'].apply(lambda x: x - pd.Timedelta(x.day % 3, unit='D'))
    # Group the data by the new 'date_3days' column and count the number of items in each group.
    time_slice = df.groupby('date_3days').size().tolist()

    corpus_filepath = config.MODELS_FOLDER + 'dtm/2023-05-02_13-35/corpus.mm'
    dictionary_filepath = config.MODELS_FOLDER + 'dtm/2023-05-02_13-35/dictionary.pkl'
    # Load the saved corpus.
    corpus = MmCorpus(corpus_filepath)
    # Load the saved dictionary.
    dictionary = Dictionary.load(dictionary_filepath)

    # Create a new DataFrame from the groupby result.
    time_slice_df = df.groupby('date_3days').size().reset_index(name='count')
    # Add a column for the progressive number of the time slice.
    time_slice_df['progressive_number'] = range(1, len(time_slice_df) + 1)

    # Create a custom x-axis label for each time slice with the progressive number and date range.
    def create_x_axis_label(row):
        start_date = row['date_3days']
        end_date = start_date + pd.Timedelta(2, unit='D')
        return f"{row['progressive_number']} ({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})"

    time_slice_df['x_axis_label'] = time_slice_df.apply(create_x_axis_label, axis=1)

    # Plot the bar chart.
    fig, ax = plt.subplots()
    ax.bar(time_slice_df['x_axis_label'], time_slice_df['count'])

    # Rotate the x-axis labels to fit properly.
    plt.xticks(rotation=90)

    # Set the axis labels and title.
    ax.set_xlabel('Time Slices')
    ax.set_ylabel('Number of toots')
    ax.set_title('Number of toots per Time Slice')

    # Adjust the bottom margin to allow more space for x-axis labels.
    plt.subplots_adjust(bottom=0.5)

    # Display the plot.
    plt.show()