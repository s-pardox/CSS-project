import pandas as pd
import numpy as np
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mastodon.settings import config
from mastodon.utils import data_mng

DATA_FILE = config.DATA_FOLDER + 'data.csv'

def do() -> None:
    pd.set_option('display.max_rows', 5000)
    pd.set_option('display.max_columns', 5000)
    pd.set_option('display.width', 10000)
    pd.set_option('display.max_colwidth', 1000)

    df = data_mng.load_from_csv_to_df(DATA_FILE)

    df_with_sentiment = analyze_sentiment(df)
    print(df_with_sentiment['sentiment_category'].value_counts())

    # plot_sentiment(df_with_sentiment)
    plot_sentiment_4(df_with_sentiment)

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform sentiment analysis on the 'filtered_content' column of a given dataframe
    and return the dataframe with additional columns for sentiment polarity and sentiment category.

    :param df: The input dataframe with a column named 'filtered_content' containing preprocessed toot text.
    :return: The input dataframe with additional columns for sentiment polarity and sentiment category.
    """

    def get_sentiment(text):
        return TextBlob(text).sentiment.polarity

    df['sentiment'] = df['filtered_content'].apply(get_sentiment)

    def sentiment_category(sentiment):
        if sentiment > 0:
            return 'positive'
        elif sentiment < 0:
            return 'negative'
        else:
            return 'neutral'

    df['sentiment_category'] = df['sentiment'].apply(sentiment_category)

    return df

def plot_sentiment(df: pd.DataFrame) -> None:
    """
    Generate a plot of sentiment analysis results over time from a dataframe containing
    'date_3days', 'sentiment', and 'sentiment_category' columns, aggregated on a 3-day basis.

    :param df: The input dataframe with 'date_3days', 'sentiment', and 'sentiment_category' columns.
    :return: None
    """

    # Group by 'date_3days' and 'sentiment_category', then calculate the mean sentiment.
    df_agg = df.groupby(['date_3days', 'sentiment_category'], as_index=False).agg({'sentiment': np.mean})

    # Pivot the aggregated dataframe to have sentiment categories as columns.
    df_pivot = df_agg.pivot_table(index='date_3days', columns='sentiment_category', values='sentiment')

    # Create a line plot with sentiment polarity over time.
    plt.figure(figsize=(12, 6))
    plt.plot(df_pivot.index, df_pivot['positive'], label='positive', linestyle='-', marker='o')
    plt.plot(df_pivot.index, df_pivot['neutral'], label='neutral', linestyle='-', marker='o')
    plt.plot(df_pivot.index, df_pivot['negative'], label='negative', linestyle='-', marker='o')

    plt.title('Sentiment Analysis Over Time (3-day average)')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Polarity')
    plt.legend(title='Sentiment Category')
    plt.grid(True)
    plt.show()

def plot_sentiment_2(df: pd.DataFrame) -> None:
    """
    Generate a plot of sentiment analysis results over time from a dataframe containing
    'created_at', 'sentiment', and 'sentiment_category' columns, aggregated on a daily basis.

    :param df: The input dataframe with 'created_at', 'sentiment', and 'sentiment_category' columns.
    :return: None
    """

    # Convert 'created_at' to datetime type
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Set 'created_at' as index
    df.set_index('created_at', inplace=True)

    # Create separate dataframes for each sentiment category
    positive_df = df[df['sentiment_category'] == 'positive']
    neutral_df = df[df['sentiment_category'] == 'neutral']
    negative_df = df[df['sentiment_category'] == 'negative']

    # Resample and aggregate data by daily intervals for each sentiment category
    positive_agg = positive_df['sentiment'].resample('D').mean()
    neutral_agg = neutral_df['sentiment'].resample('D').mean()
    negative_agg = negative_df['sentiment'].resample('D').mean()

    # Create a line plot with sentiment polarity over time
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(positive_agg.index, positive_agg, label='positive', linestyle='-', marker='o')
    ax.plot(neutral_agg.index, neutral_agg, label='neutral', linestyle='-', marker='o')
    ax.plot(negative_agg.index, negative_agg, label='negative', linestyle='-', marker='o')

    ax.set_title("Sentiment Analysis Over Time (daily average)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment Polarity")
    ax.legend(title='Sentiment Category')
    ax.grid(True)

    # Set proper date formatting
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.show()

def plot_sentiment_3(df: pd.DataFrame) -> None:
    """
    Generate a stacked bar chart of sentiment analysis results over time from a dataframe containing
    'created_at', 'sentiment', and 'sentiment_category' columns, aggregated on a daily basis.

    :param df: The input dataframe with 'created_at', 'sentiment', and 'sentiment_category' columns.
    :return: None
    """

    # Convert 'created_at' to datetime type
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Count the sentiment categories for each day
    df_count = df.groupby([pd.Grouper(key='created_at', freq='D'), 'sentiment_category']).size().reset_index(name='count')

    # Pivot the count dataframe to have sentiment categories as columns
    df_pivot = df_count.pivot_table(index='created_at', columns='sentiment_category', values='count', fill_value=0)

    # Normalize the count of sentiment categories for each day
    df_pivot = df_pivot.div(df_pivot.sum(axis=1), axis=0)

    # Create a stacked bar chart with sentiment category proportions over time
    fig, ax = plt.subplots(figsize=(12, 6))

    df_pivot.plot.bar(stacked=True, ax=ax)

    ax.set_title("Sentiment Analysis Over Time (daily proportions)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment Category Proportion")
    ax.legend(title='Sentiment Category')

    # Set proper date formatting
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.show()

def plot_sentiment_4(df: pd.DataFrame) -> None:
    """
    Generate an area plot of sentiment analysis results over time from a dataframe containing
    'created_at', 'sentiment', and 'sentiment_category' columns, aggregated on a 3-day basis.

    :param df: The input dataframe with 'created_at', 'sentiment', and 'sentiment_category' columns.
    :return: None
    """

    # Convert 'created_at' to datetime type
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Count the sentiment categories for each day
    df_count = df.groupby([pd.Grouper(key='created_at', freq='3D'), 'sentiment_category']).size().reset_index(name='count')

    # Pivot the count dataframe to have sentiment categories as columns
    df_pivot = df_count.pivot_table(index='created_at', columns='sentiment_category', values='count', fill_value=0)

    # Normalize the count of sentiment categories for each 3-day interval
    df_pivot = df_pivot.div(df_pivot.sum(axis=1), axis=0)

    # Create an area plot with sentiment category proportions over time
    fig, ax = plt.subplots(figsize=(12, 6))

    df_pivot.plot.area(ax=ax)

    ax.set_title("Sentiment Analysis Over Time (3-day proportions)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment Category Proportion")
    ax.legend(title='Sentiment Category')

    # Set proper date formatting
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.show()