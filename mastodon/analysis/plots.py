import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import DateFormatter, WeekdayLocator
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Union

from mastodon.settings import config
from mastodon.utils import data_mng


def RQ1(filepath: str, events_filepath: str = None, additional_filepath: str = None) -> None:
    """
    Function to generate plots mostly related to the first research question: "How do key events related to
    Elon Musk's acquisition of Twitter correlate with surge in user registrations on the alternative
    platform Mastodon, potentially signifying user migration trends?"

    :param filepath: str: The file path to the main dataset in CSV format.
    :param events_filepath: str: The file path to the events dataset in JSON format, defaults to None.
    :param additional_filepath: str: The file path to the additional dataset in CSV format, defaults to None.

    :return: None
    """

    df_orig = data_mng.load_from_csv_to_df(filepath)

    plot_daily_increment_v1(df_orig.copy())
    plot_daily_increment_v2(df_orig.copy())

    events_df = data_mng.load_from_json_to_df(events_filepath) if events_filepath is not None else None
    plot_smoothed_daily_increment_with_events(df_orig.copy(), events_df)

    additional_df_orig = data_mng.load_from_csv_to_df(additional_filepath)

    plot_users_daily_increment_and_hashtag_usage(df_orig.copy(), additional_df_orig.copy(), start_date='2022-10-20',
                                                 end_date='2023-01-01')
    plot_users_daily_increment_and_hashtag_usage_with_events(df_orig.copy(), additional_df_orig.copy(),
                                                             start_date='2022-10-20', end_date='2023-01-01',
                                                             events_df=events_df)

def RQ2(filepath: str, events_filepath: str = None) -> None:
    """
    Function to generate plots and calculate statistics mostly related to the second research question: "How does
    the activity of newcomers, as evidenced by their engagement in public toots within the most representative
    Mastodon federated instance, correlate with the daily enrollment rate of new users on the platform?"

    :param filepath: str: The file path to the main dataset in CSV format.
    :param events_filepath: str: The file path to the events dataset in JSON format, defaults to None.

    :return: None
    """

    df = data_mng.load_from_csv_to_df(filepath, index_col=None)

    events_df = data_mng.load_from_json_to_df(events_filepath) if events_filepath is not None else None
    plot_user_engagement_over_time(df.copy(), events_df)

    plot_engagement_ratio_and_trend(df.copy())
    calculate_and_show_engagement_statistics(df)


def RQ3(filepath: str) -> None:
    """
    Function to generate plots related to the third research question: "What are the primary themes and patterns
    evident in the reasons users provide for migrating from Twitter to Mastodon?"

    :param filepath: str: The file path to the main dataset in CSV format.

    :return: None
    """

    df = data_mng.load_from_csv_to_df(filepath, index_col=None)

    plot_frequency_by_date(df, start_date='2022-10-18', end_date='2023-05-15')
    plot_freq_instances(df, end_date='2023-05-15')
    plot_freq_words(df)

def set_pandas_options() -> None:
    """
    Sets specific display options for pandas DataFrame.
    """
    pd.set_option('display.max_rows', 5000)
    pd.set_option('display.max_columns', 5000)
    pd.set_option('display.width', 10000)
    pd.set_option('display.max_colwidth', 1000)

def plot_users_daily_increment_and_hashtag_usage(df1: pd.DataFrame, df2: pd.DataFrame,
                                                 start_date: str = None, end_date: str = None,
                                                 window_size: int = 3) -> None:
    """
    Plots the moving average of daily increments in 'total_accounts' column of the first DataFrame, and the smoothed sum of
    the frequency of tweets over time for all hashtags in the second DataFrame.

    :param df1: pd.DataFrame, The DataFrame containing the 'total_accounts' data.
    :param df2: pd.DataFrame, The DataFrame containing the 'hashtag' data.
    :param start_date: str, optional, Start date for time period selection (default: None).
    :param end_date: str, optional, End date for time period selection (default: None).
    :param window_size: int, optional, Size of the moving window for the rolling mean calculation (default: 7).
    """

    # Ensure start_date and end_date are of datetime type.
    if start_date is not None:
        start_date = pd.to_datetime(start_date).tz_localize('UTC')
    if end_date is not None:
        end_date = pd.to_datetime(end_date).tz_localize('UTC')

    # Prepare df1.
    df1['created_at'] = pd.to_datetime(df1['created_at'])
    df1.set_index('created_at', inplace=True)

    # Filter df1 based on start_date and end_date
    if start_date is not None:
        df1 = df1[df1.index >= start_date]
    if end_date is not None:
        df1 = df1[df1.index <= end_date]

    # Resample the data to daily frequency, calculating the mean value for each day.
    daily_df1 = df1.resample('D').mean()

    # Calculate daily increment in 'total_accounts' column.
    daily_df1['daily_increment'] = daily_df1['total_accounts'].diff()

    # Compute the moving average of 'daily_increment' over a window size defined by 'window_size' parameter.
    daily_df1['moving_average'] = daily_df1['daily_increment'].rolling(window=window_size).mean()

    # Prepare df2.
    df2['date'] = pd.to_datetime(df2['date']).dt.tz_localize('UTC')

    # Filter df2 based on start_date and end_date
    if start_date is not None:
        df2 = df2[df2['date'] >= start_date]
    if end_date is not None:
        df2 = df2[df2['date'] <= end_date]

    df2 = df2.groupby(['date', 'hashtag']).size().reset_index(name='count')
    df_pivot = df2.pivot(index='date', columns='hashtag', values='count').fillna(0)

    # Calculate the sum of posts over all hashtags for each day.
    df_pivot['total_posts'] = df_pivot.sum(axis=1).rolling(window=window_size).mean()

    """BEGIN: Cross-correlation calculation attempt; not working as expected.
    # Ensure the data is stationary by differencing.
    stationary_df1 = daily_df1['moving_average'].diff().dropna()
    stationary_df2 = df_pivot['total_posts'].diff().dropna()

    print(stationary_df1)
    print(stationary_df2)

    # Compute the cross-correlation.
    cross_correlation = np.correlate(stationary_df1, stationary_df2, mode='same')

    # Compute the lag for the maximum cross-correlation.
    lag = np.argmax(cross_correlation)

    print(f"The lag for maximum cross-correlation is: {lag}")
    END"""

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the moving average from df1.
    color1 = 'tab:blue'
    ax1.plot(daily_df1.index, daily_df1['moving_average'], color=color1, label="Moving Avg of Daily Increment in Total Accounts")
    ax1.set_ylabel('Moving Avg of Increment in Total Accounts', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.tick_params(axis='x', labelrotation=45)

    # Set the x-axis limits to make sure the curves start exactly at the left border.
    ax1.set_xlim(left=start_date, right=end_date)

    # Set the x-axis date ticks to weekly frequency (or any other frequency you prefer).
    ax1.xaxis.set_major_locator(WeekdayLocator(interval=1))  # Change the interval parameter to adjust the tick frequency
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

    # Plot the total posts from df2.
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.plot(df_pivot.index, df_pivot['total_posts'], color=color2, label='Smoothed Total Posts')
    ax2.set_ylabel('Smoothed Total Posts', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Set the x-axis limit to include all dates from start_date to end_date.
    if start_date is not None and end_date is not None:
        ax1.set_xlim(start_date, end_date)
        ax2.set_xlim(start_date, end_date)

    # Set the title and x-label.
    plt.title('Comparison of Total Accounts Increment and Total Posts Over Time')
    plt.xlabel('Date')

    fig.tight_layout()
    plt.show()

def plot_users_daily_increment_and_hashtag_usage_with_events(df1: pd.DataFrame, df2: pd.DataFrame, events_df: pd.DataFrame,
                                                             start_date: str = None, end_date: str = None,
                                                             window_size: int = 3) -> None:
    """
    Plots the moving average of daily increments in 'total_accounts' column of the first DataFrame, the smoothed sum of
    the frequency of tweets over time for all hashtags in the second DataFrame, and annotations for events.

    :param df1: pd.DataFrame, The DataFrame containing the 'total_accounts' data.
    :param df2: pd.DataFrame, The DataFrame containing the 'hashtag' data.
    :param events_df: pd.DataFrame, The DataFrame containing the events.
    :param start_date: str, optional, Start date for time period selection (default: None).
    :param end_date: str, optional, End date for time period selection (default: None).
    :param window_size: int, optional, Size of the moving window for the rolling mean calculation (default: 7).
    """

    def thousands(x, pos):
        """
        Formats the y-axis ticks in thousands with 'k' as a suffix.

        :param x: The tick value.
        :param pos: The tick position. This argument is needed for compatibility with the FuncFormatter.
        :return: The formatted tick label.
        """
        return '%1.0fk' % (x * 1e-3)

    formatter = FuncFormatter(thousands)

    if start_date is not None:
        start_date = pd.to_datetime(start_date).tz_localize('UTC')
    if end_date is not None:
        end_date = pd.to_datetime(end_date).tz_localize('UTC')

    df1['created_at'] = pd.to_datetime(df1['created_at'])
    df1.set_index('created_at', inplace=True)

    if start_date is not None:
        df1 = df1[df1.index >= start_date]
    if end_date is not None:
        df1 = df1[df1.index <= end_date]

    daily_df1 = df1.resample('D').mean()

    daily_df1['daily_increment'] = daily_df1['total_accounts'].diff()

    daily_df1['moving_average'] = daily_df1['daily_increment'].rolling(window=window_size).mean()

    df2['date'] = pd.to_datetime(df2['date']).dt.tz_localize('UTC')

    if start_date is not None:
        df2 = df2[df2['date'] >= start_date]
    if end_date is not None:
        df2 = df2[df2['date'] <= end_date]

    df2 = df2.groupby(['date', 'hashtag']).size().reset_index(name='count')
    df_pivot = df2.pivot(index='date', columns='hashtag', values='count').fillna(0)

    df_pivot['total_posts'] = df_pivot.sum(axis=1).rolling(window=window_size).mean()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.plot(daily_df1.index, daily_df1['moving_average'], color=color1,
             label="Moving average of daily increment in total accounts")
    ax1.set_ylabel('Moving average of daily increment in total accounts', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.yaxis.set_major_formatter(formatter)

    ax2 = ax1.twinx()

    color2 = 'tab:orange'
    ax2.plot(df_pivot.index, df_pivot['total_posts'], color=color2, label="Sum of toots with #mastodonmigration and #twittermigration hashtags")
    ax2.set_ylabel('Sum of toots with #mastodonmigration and #twittermigration hashtags', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.autofmt_xdate()
    ax1.xaxis.set_major_locator(WeekdayLocator(byweekday=mdates.MO))
    ax1.xaxis.set_major_formatter(DateFormatter('%b %d'))

    # Event plotting

    # Determine the maximum y value in the 'moving_average' column.
    y_max = daily_df1['moving_average'].max()

    # Define the y-coordinate position for labels, setting it 3% above the maximum y value.
    y_label_position = y_max - 0.03 * y_max

    # Ensuring the event_date is Timestamp.
    events_df['date'] = pd.to_datetime(events_df['date']).dt.tz_localize('UTC')

    # Loop over the events and add labels with adjusted y-coordinate.
    legend_labels = []
    for idx, event in events_df.iterrows():
        event_no = event['event_no']
        event_date = event['date']
        legend_label = event['legend_label']

        if start_date <= event_date <= end_date:
            ax1.axvline(x=event_date, color='gray', linestyle='--', alpha=0.6)

            # Adjust the y-coordinate position for labels
            text_y = y_label_position
            ax1.text(event_date, text_y, 'E' + event_no, verticalalignment='center', fontsize=12)

            # Add the event summary to the legend labels.
            legend_labels.append(f'Event nr. {event_no} - {event_date.strftime("%Y-%m-%d")}: {legend_label}')

    # Set the y limit of the plot to accommodate the labels.
    ax1.set_ylim(0, y_max)

    # Add a legend on the top right of the plot.
    # ax1.legend(legend_labels, loc='upper right', bbox_to_anchor=(1.0, 1.0))

    plt.title('Comparison of total accounts increment and toots with #mastodonmigration and #twittermigration hashtags over time')

    fig.tight_layout()
    plt.show()

def plot_frequency_by_date(df: pd.DataFrame, start_date: str = None, end_date: str = None) -> None:
    """
    Plots the frequency of tweets over time for each hashtag.

    :param df: pd.DataFrame, The DataFrame containing the filtered content.
    :param start_date: str, optional, Start date for time period selection (default: None).
    :param end_date: str, optional, End date for time period selection (default: None).

    :return: None

    Plots a line graph showing the volume of tweets over time for each hashtag. The x-axis represents
    the dates, and the y-axis represents the number of tweets. The plot is based on the daily frequency
    of tweets. The time period for the plot can be adjusted by providing a start date and/or end date.
    """
    # Convert the 'date' column to datetime format.
    df['date'] = pd.to_datetime(df['date'])

    # Filter the DataFrame based on the provided date range.
    if start_date is not None:
        df = df[df['date'] >= start_date]
    if end_date is not None:
        df = df[df['date'] <= end_date]

    # Group by date and hashtag, then reset index.
    df = df.groupby(['date', 'hashtag']).size().reset_index(name='count')

    # Pivot the DataFrame to have a column for each hashtag's count.
    df_pivot = df.pivot(index='date', columns='hashtag', values='count').fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot a line for each hashtag.
    for column in df_pivot.columns:
        ax.plot(df_pivot.index, df_pivot[column], label=column)

    ax.set(title='Volume of toots over time on Mastodon', xlabel='Date', ylabel='# of posts')
    ax.tick_params(axis='x', labelrotation=45)  # Set rotation to 45 degrees

    # Define the date format
    date_form = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_form)

    # Ensure a major tick for each week using (interval=1)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    # Set x-axis limits to start_date and end_date
    if start_date:
        ax.set_xlim(left=pd.to_datetime(start_date))
    if end_date:
        ax.set_xlim(right=pd.to_datetime(end_date))

    # Add a legend to the plot.
    ax.legend()

    plt.tight_layout()  # Adjust subplot parameters to give specified padding.
    plt.show()

def plot_moving_AVG(df: pd.DataFrame) -> None:
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.groupby(pd.Grouper(key='created_at', freq='D')).size().reset_index(name='count')
    df.set_index('created_at', inplace=True)

    # Calculate the rolling average with a window size of 3 days.
    rolling_avg = df.rolling(window='3D').mean()

    # Plot the frequency and rolling average on the same plot.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df, label='Frequency')
    ax.plot(rolling_avg, label='Rolling Average (3 days)')
    ax.set(title='Frequency and Rolling Average of #TwitterMigration toots', xlabel='Date', ylabel='Count')
    ax.legend()
    plt.show()

def plot_freq_instances(df: pd.DataFrame, min_toots: int = None, max_bars: int = 10, start_date: str = None,
                        end_date: str = None) -> None:
    """
    Plot the frequency of instances in a DataFrame based on the number of toots, grouped by hashtag.

    :param df: pd.DataFrame, The DataFrame containing the filtered content.
    :param min_toots: int, optional, Minimum number of toots required to include an instance (default: None).
    :param max_bars: int, optional, Maximum number of instances to display in the plot (default: 10).
    :param start_date: str, optional, Start date for time period selection (default: None).
    :param end_date: str, optional, End date for time period selection (default: None).

    :return: None

    Plots a bar graph showing the frequency of instances based on the number of toots they have. The plot
    includes the top instances based on either the provided maximum number of bars or the minimum number
    of toots threshold. The x-axis represents the instance names, and the y-axis represents the number
    of toots.
    """
    # Convert the 'date' column to datetime format.
    df['date'] = pd.to_datetime(df['date'])

    # Filter the DataFrame based on the provided date range.
    if start_date is not None:
        df = df[df['date'] >= start_date]
    if end_date is not None:
        df = df[df['date'] <= end_date]

    # Count the number of toots per instance for each hashtag.
    instance_counts = df.groupby(['instance', 'hashtag']).size().unstack(fill_value=0)

    # If min_toots is set, keep instances with toots more than the threshold.
    if min_toots is not None:
        instance_counts = instance_counts[instance_counts.sum(axis=1) >= min_toots]

    # Sum across the rows and sort in descending order.
    instance_counts['total'] = instance_counts.sum(axis=1)
    instance_counts = instance_counts.sort_values('total', ascending=False)

    # Drop the 'total' column as we no longer need it.
    instance_counts = instance_counts.drop(columns='total')

    # If max_bars is set, keep the top instances.
    if max_bars is not None:
        instance_counts = instance_counts.head(max_bars)

    # Create a bar plot of the final counts.
    fig, ax = plt.subplots()
    instance_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Instance name')
    ax.set_ylabel('# toots')

    # Set the title of the plot
    ax.set_title('Number of toots per instance for each hashtag, from the beginning until 2023-05-15')

    # Adjust subplot parameters to move the graph up and give more space to x-axis labels.
    fig.subplots_adjust(bottom=0.25)

    plt.show()

def plot_freq_langs(df: pd.DataFrame) -> None:
    df_freq = df.groupby(pd.Grouper(key='lang')).size().reset_index(name='count')
    plt.bar(df_freq['lang'], df_freq['count'])

    plt.xlabel('Language')
    plt.ylabel('Count')
    plt.show()

def plot_freq_words(df: pd.DataFrame) -> None:
    """
    Plots the frequency of words from the 'filtered_content' column of the DataFrame.

    The 'filtered_content' is expected to have been pre-processed using a Term Frequency-Inverse Document Frequency
    (TF-IDF) filtering procedure. This function tokenizes the filtered content, counts the frequency of each word,
    and then plots a bar chart of the top 20 most frequent words.

    :param df: pd.DataFrame, The DataFrame containing the 'filtered_content' column.
    :return: None
    """
    tokenized_data = [text.split() for text in df['filtered_content']]

    # Convert tokenized data into a list of words.
    words = [word for doc in tokenized_data for word in doc]

    # Count the frequency of each word.
    word_freq = Counter(words)

    # Create a bar plot of the top 20 most frequent words.
    top_words = word_freq.most_common(20)
    x, y = zip(*top_words)
    plt.bar(x, y)
    plt.title('Top 20 Most Frequent Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_freq_hashtags(df: pd.DataFrame) -> None:
    tokenized_data = [text.split() for text in df['cleaned_content']]

    # Convert tokenized data into a list of words.
    words = [word for doc in tokenized_data for word in doc]

    # Count the frequency of each word.
    word_freq = Counter(words)

    # Select the most frequent words that begin with '#'.
    hashtag_freq = {k: v for k, v in word_freq.items() if k.startswith('#')}

    # Create a bar plot of the top 20 most frequent hashtags.
    top_hashtags = dict(sorted(hashtag_freq.items(), key=lambda item: item[1], reverse=True)[:20])
    try:
        x, y = zip(*top_hashtags.items())
    except ValueError:
        print('Are you sure there are any #hashtag?')

    plt.bar(x, y)
    plt.title('Top 20 Most Frequent Hashtags')
    plt.xlabel('Hashtags')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_top_n_terms(df, N=20, stop_words=None):
    """
    This function takes a DataFrame and an integer N as inputs,
    and creates a bar chart of the top N terms with the highest average TF-IDF scores.

    :param df: The DataFrame containing the filtered content.
    :param N: The number of top terms to display in the bar chart (default: 20).
    :param stop_words: A list of stop words to be used by the TfidfVectorizer (default: None).
    """

    # Create a TfidfVectorizer instance with the stop_words parameter.
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stop_words)

    # Fit the vectorizer on the cleaned_content column to get the vocabulary.
    vectorizer.fit(df['cleaned_content'])

    # Calculate average TF-IDF scores.
    tfidf_matrix = vectorizer.transform(df['cleaned_content'])
    avg_tfidf_scores = tfidf_matrix.mean(axis=0).tolist()[0]
    term_scores = {term: score for term, score in zip(vectorizer.get_feature_names_out(), avg_tfidf_scores)}

    # Get the top N terms.
    top_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:N]

    # Create the bar chart
    terms, scores = zip(*top_terms)
    indices = np.arange(len(terms))
    plt.figure(figsize=(10, 5))
    plt.bar(indices, scores, align='center')
    plt.xticks(indices, terms, rotation=45, ha='right')
    plt.xlabel('Top Terms')
    plt.ylabel('Average TF-IDF Score')
    plt.title('Top {} Terms based on Average TF-IDF Scores'.format(N))
    plt.tight_layout()
    plt.show()

def plot_daily_increment_v1(df: pd.DataFrame) -> None:
    """
    Plots the daily increment in 'total_accounts' column of the provided DataFrame as a time series.

    :param df: The DataFrame to plot data from.
    """
    # Convert 'created_at' to datetime if it isn't already.
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Set 'created_at' as the index of the DataFrame to facilitate time-series plotting.
    df.set_index('created_at', inplace=True)

    # Compute the daily increment in 'total_accounts'.
    df['daily_increment'] = df['total_accounts'].diff()

    # Replace negative increments with 0.
    df['daily_increment'] = df['daily_increment'].apply(lambda x: max(x, 0))

    # Plot 'daily_increment' column.
    plt.figure(figsize=(10, 5))  # Set the figure size (optional)
    df['daily_increment'].plot(kind='line')

    # Provide labels for clarity.
    plt.title('Daily Increment in Total Accounts')
    plt.xlabel('Date')
    plt.ylabel('Increment in Total Accounts')

    # Show the plot.
    plt.show()

def plot_daily_increment_v2(df: pd.DataFrame) -> None:
    """
    Plots the 'last_day' column of the provided DataFrame as a time series.

    :param df: The DataFrame to plot data from.
    """
    # Convert 'created_at' to datetime if it isn't already.
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Set 'created_at' as the index of the DataFrame to facilitate time-series plotting.
    df.set_index('created_at', inplace=True)

    # Plot 'last_day' column.
    # Set the figure size (optional).
    plt.figure(figsize=(10, 5))
    df['daily_increment'].plot(kind='line')

    # Provide labels for clarity.
    plt.title('Daily Increment in Total Accounts')
    plt.xlabel('Date')
    plt.ylabel('Increment in Total Accounts')

    # Show the plot.
    plt.show()

def plot_smoothed_daily_increment_with_events(df: pd.DataFrame, events_df: pd.DataFrame, window_size: int = 24) -> None:
    """
    Plots the smoothed daily increment in total accounts using a rolling average.

    :param df: DataFrame containing the data.
    :param events_df: DataFrame containing the events. It has 'date', 'summary', 'arrow_dx', 'arrow_dy', 'label_fontsize', and 'label_color' columns.
    :param window_size: Window size for the rolling average (default: 24 hours).
    """

    # Define a function to format y-axis ticks into thousands ('k').
    def thousands(x, pos):
        """
        Formats the y-axis ticks in thousands with 'k' as a suffix.

        :param x: The tick value.
        :param pos: The tick position. This argument is needed for compatibility with the FuncFormatter.
        :return: The formatted tick label.
        """
        return '%1.0fk' % (x * 1e-3)

    # Instantiate the formatter object using our custom function.
    formatter = FuncFormatter(thousands)

    ### Data processing.

    # Ensure 'created_at' column is of datetime type.
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Set 'created_at' column as index for easier data manipulation.
    df.set_index('created_at', inplace=True)

    # Resample the data to daily frequency, calculating the mean value for each day.
    daily_df = df.resample('D').mean()

    # Calculate daily increment in 'total_accounts' column.
    # We should have a correct daily_increment column, yet.
    # daily_df['daily_increment'] = daily_df['total_accounts'].diff()

    # Compute the moving average of 'daily_increment' over a window size defined by 'window_size' parameter.
    daily_df['moving_average'] = daily_df['daily_increment'].rolling(window=window_size).mean()

    ### Creating the plot.

    # Instantiate a new figure and axes.
    fig, ax = plt.subplots()

    # Plot the moving average line on the axes.
    daily_df['moving_average'].plot(ax=ax, color='black')

    ### Setting up labels and arrows.

    # Determine the maximum y value in the 'moving_average' column.
    y_max = daily_df['moving_average'].max()

    # Define the y-coordinate position for labels, setting it 10% above the maximum y value.
    y_label_position = y_max + 0.10 * y_max

    ### Loop over all events in the 'events_df' DataFrame.

    # Iterate over the events dataframe and create annotations for each event.
    for idx, event in events_df.iterrows():
        # Convert the date string to datetime format.
        date = pd.to_datetime(event['date'])

        # Get the summary text from the event.
        summary = event['summary']

        # Read the dx and dy values for arrow displacement.
        arrow_dx = float(event['arrow_dx'])
        arrow_dy = float(event['arrow_dy'])

        # Get the font size and color for the label.
        label_fontsize = int(event['label_fontsize'])
        label_color = event['label_color']

        # Draw a vertical line on the specified date.
        ax.axvline(pd.to_datetime(date), linestyle='--', color='darkred')

        # The coordinates where the text will be placed.
        # We first convert the date to a numerical format understandable by matplotlib.
        arrow_x = mdates.date2num(date)

        # The x-coordinate for text is same as the arrow's x-coordinate
        text_x = arrow_x + arrow_dx

        # The y-coordinate for text is slightly below the y_label_position.
        y_offset = int(event['y_offset'])
        text_y = y_label_position - 0.05 * y_max + y_offset

        # The arrow points to a position slightly below the text.
        arrow_y = text_y - arrow_dy

        # Adjust the arrow's origin depending on whether it's on the left or right side of the vertical line.
        if arrow_dx < 0:  # If on the left, arrow comes from bottom-center of the text.
            ha, va, connectionstyle = 'center', 'top', 'arc3,rad=.2'
        else:  # If on the right, arrow comes from bottom-center of the text.
            ha, va, connectionstyle = 'center', 'top', 'arc3,rad=-.2'

        # Create the annotation (label with an arrow).
        ax.annotate(summary,
                    # Exact coordinates where the arrow has to point.
                    xy=(arrow_x, arrow_y),
                    # Text coordinates.
                    xytext=(text_x, text_y),
                    xycoords='data',
                    textcoords='data',
                    # Text alignment.
                    ha=ha, va=va,
                    fontsize=label_fontsize, color=label_color,
                    arrowprops=dict(arrowstyle='->', color='darkred',
                                    connectionstyle=connectionstyle))

    # Set the y limit of the plot to accommodate the labels.
    plt.ylim(0, y_label_position)

    # Apply the custom formatter to each y-tick label.
    ax.yaxis.set_major_formatter(formatter)

    ### Setting up the x-axis.

    # Define the lower limit of the x-axis to be the minimum date in the DataFrame.
    start_date = pd.to_datetime('2022-02-01')
    # Set the limits of the x-axis.
    ax.set_xlim([start_date, daily_df.index.max()])

    # Add labels to the x-axis and y-axis.
    ax.set_xlabel("Month")
    ax.set_ylabel("Daily Mastodon accounts increment (smoothed)")

    # Format x-tick labels as 3-letter month name.
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # Set x-axis major ticks to monthly interval, on the 1st day of the month.
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    ### Add top year labels.

    # Plot vertical lines and year labels at the start of each year.
    years_to_plot = list(range(df.index.year.min(), df.index.year.max() + 1))
    for year in years_to_plot:
        # Plot a vertical line at the start of the year.
        ax.axvline(pd.to_datetime(f'{year}-01-01'), linestyle='--', color='gray', alpha=0.5)

    # Get years from the index of the DataFrame.
    years = daily_df.index.year.unique()

    # Add text annotations for each year at the top of the plot.
    for i, year in enumerate(years):
        # Set x-offset to 100 pixels for the first label, 0 for others.
        x_offset = 40 if i == 0 else 10
        ax.text(mdates.date2num(pd.to_datetime(f'{year}-01-01')) + x_offset, 0.96 * y_label_position, str(year),
                verticalalignment='center', fontsize=12, ha='center', color='gray', va='bottom')

    plt.title('Mastodon daily user accounts increment and top 10 significant events involving Elon Musk')

    plt.show()

def plot_user_engagement_over_time(df: pd.DataFrame,
                                              events_df: pd.DataFrame = None,
                                              window_size: int = 7) -> None:
    """
    Plots the smoothed daily increment in total accounts using a rolling average.

    :param df: DataFrame containing the data.
    :param events_df: DataFrame containing the events. It has 'date', 'summary',
    'arrow_dx', 'arrow_dy', 'label_fontsize', and 'label_color' columns.
    :param window_size: Window size for the rolling average (default: 7 days).
    """
    # Make sure 'enrollment_day' is datetime.
    df['enrollment_day'] = pd.to_datetime(df['enrollment_day'])
    if events_df is not None:
        events_df['date'] = pd.to_datetime(events_df['date'])

    # Set 'enrollment_day' as index for smoothing operation.
    df.set_index('enrollment_day', inplace=True)

    # Create rolling average columns.
    df_smooth = df.rolling(window=window_size).mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data.
    ax.plot(df_smooth.index, df_smooth['incremental_new_users'], label='Incremental new users')
    ax.plot(df_smooth.index, df_smooth['toots_on_enrollment_day'], label='New users active on the enrollment day')
    ax.plot(df_smooth.index, df_smooth['toots_in_next_3_days'], label='New users active in next 3 days')
    ax.plot(df_smooth.index, df_smooth['toots_days_4_to_7'], label='New users active between the 4th and 7th days')

    # Plot event arrows.
    if events_df is not None:
        for i, event in events_df.iterrows():
            ax.annotate(event['summary'],
                        xy=(event['date'], df.loc[event['date'], 'incremental_new_users']),
                        xytext=(event['arrow_dx'], event['arrow_dy']),
                        textcoords='offset points',
                        arrowprops=dict(facecolor='red', shrink=0.05),
                        fontsize=event['label_fontsize'],
                        color=event['label_color'])

    # Formatting date.
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.title('Smoothed Daily Increment in Total Accounts')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_engagement_ratio_and_trend(df: pd.DataFrame) -> None:
    """
    This function takes a pandas DataFrame as input, computes an 'engagement_ratio' column as the ratio of
    'distinct_tooting_new_users' to 'incremental_users', and creates a line plot of this engagement ratio over time.
    It also plots the trend line of 'incremental_users' on the same plot. The plot is displayed on the screen.

    The DataFrame should have the following columns:
    'enrollment_day': Date of user enrollment. Expected to be of datetime type.
    'incremental_users': Day-to-day growth of the users on the platform. Expected to be of int type.
    'distinct_tooting_new_users': Count of distinct new users who made toots. Expected to be of int type.

    :param df: DataFrame containing the data.
    :return: None
    """
    df['engagement_ratio'] = np.where(df['incremental_new_users'] != 0,
                                      df['toots_on_enrollment_day'] / df['incremental_new_users'], np.nan)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Enrollment Day')
    ax1.set_ylabel('Engagement Ratio', color=color)
    ax1.plot(df['enrollment_day'], df['engagement_ratio'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Incremental Users', color=color)
    ax2.plot(df['enrollment_day'], df['incremental_new_users'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Engagement Ratio and Incremental Users Over Time')

    # Rotate x-axis labels by 90 degrees.
    ax1.set_xticklabels(df['enrollment_day'], rotation=45, fontsize=10)

    plt.tight_layout()
    plt.show()

def calculate_and_show_engagement_statistics(df: pd.DataFrame) -> None:
    """
    This function calculates and visualizes various statistics and plots for the input DataFrame.

    The DataFrame should have the following columns:
    - 'enrollment_day': Date of user enrollment. Expected to be of datetime type.
    - 'incremental_new_users': Day-to-day growth of the users on the platform. Expected to be of int type.
    - 'toots_on_enrollment_day': Total posts by the user on the enrollment day. Expected to be of int type.
    - 'toots_in_next_3_days': Total posts by the user in the 3 days following the enrollment day. Expected to be of int type.
    - 'toots_days_4_to_7': Total posts by the user in the 4th to 7th day period after the enrollment day. Expected to be of int type.

    The function visualizes:
    1. A correlation heatmap giving an overview of how all variables are correlated with each other.
    2. A line plot visualizing how the 'incremental_new_users' change over time.
    3. A histogram of 'incremental_new_users' showing the distribution of values.

    :param df: DataFrame containing the data.
    :return: None
    """

    def create_correlation_heatmap(df: pd.DataFrame) -> None:
        """
        Creates a correlation heatmap of the given DataFrame.
        """
        correlation_matrix = df.select_dtypes(include=[np.number]).dropna().corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()

    def create_incremental_users_lineplot(df: pd.DataFrame) -> None:
        """
        Creates a line plot showing the trend of 'incremental_new_users' over time.
        """
        plt.figure(figsize=(10, 8))
        sns.lineplot(x='enrollment_day', y='incremental_new_users', data=df)
        plt.title('Incremental New Users Over Time')
        plt.xlabel('Enrollment Day')
        plt.ylabel('Incremental New Users')
        plt.xticks(rotation=45)
        plt.show()

    def create_new_users_histogram(df: pd.DataFrame) -> None:
        """
        Creates a histogram showing the distribution of 'incremental_new_users'.
        """
        plt.figure(figsize=(8, 6))
        sns.histplot(df['incremental_new_users'], kde=True, bins=30)
        plt.title('Distribution of Incremental New Users')
        plt.xlabel('Incremental New Users')
        plt.show()

    create_correlation_heatmap(df)
    create_incremental_users_lineplot(df)
    create_new_users_histogram(df)


def plot_instances_over_time_v1(csv_file_path: str, output_file: str = None, save_plot: bool = False, top_instances: int = 10) -> None:
    """
    Plot the top instances over time based on data from the CSV file.

    :param csv_file_path: (str) File path to read the CSV data.
    :param output_file: (str) Optional. File path to save the plot image. Required if save_plot is True.
    :param save_plot: (bool) Optional. If True, saves the plot to output_file. If False, displays the plot.
    :param top_instances: (int) Optional. The number of instances with the highest total user count to plot.

    This function reads instances for each day from the CSV file, selects the top instances by total user count,
    and plots these instances and their users over time. The x-axis represents the date, and the y-axis represents the users.
    Each line on the plot represents one instance.
    Depending on the value of save_plot, the plot is either saved as an image file at the specified output path or displayed.

    Note: This function assumes that the CSV file should have columns 'date', 'name', 'users'.
    """
    # Read data from csv.
    df = pd.read_csv(csv_file_path)
    df['date'] = pd.to_datetime(df['date'])

    # Get top instances by total user count.
    top_instance_totals = df.groupby('name')['users'].sum().nlargest(top_instances)

    # Select only rows corresponding to the top instances.
    df_top = df[df['name'].isin(top_instance_totals.index)]

    # Pivot the dataframe for plotting.
    df_pivot = df_top.pivot(index='date', columns='name', values='users')

    # Reorder columns by total users in descending order.
    df_pivot = df_pivot[top_instance_totals.index]

    # Create plot.
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each instance as a separate line.
    lines = df_pivot.plot(ax=ax, legend=False)

    # Customize plot appearance.
    plt.xlabel('Date')
    plt.ylabel('Users (in thousands)')
    plt.title(f'Top {top_instances} Mastodon instances over time')
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

    # Format y-axis numbers to show in K (thousands).
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/1000), ',') + 'K'))

    # Add a numbered legend with horizontal orientation below the plot.
    legend_labels = [f'{i+1}. {col}' for i, col in enumerate(df_pivot.columns)]
    # Append '*' to the first legend label.
    # legend_labels[0] = f'{legend_labels[0]}*'
    legend = plt.legend(legend_labels, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=5, borderaxespad=1.5)

    # Apply bold formatting to the first legend label
    legend.get_texts()[0].set_fontweight('bold')

    # Enlarge the plot line corresponding to "mastodon.social" instance
    lines = ax.get_lines()
    for line in lines:
        if line.get_label() == 'mastodon.social':
            line.set_linewidth(2.5)

    plt.tight_layout()

    # Save the plot as an image or display it.
    if save_plot:
        plt.savefig(output_file, bbox_inches='tight')
    else:
        plt.show()

def plot_instances_over_time_v2(csv_file_path: str, output_file: str = None, save_plot: bool = False) -> None:
    """
    Plot all instances over time based on data from the CSV file.

    :param csv_file_path: (str) File path to read the CSV data.
    :param output_file: (str) Optional. File path to save the plot image. Required if save_plot is True.
    :param save_plot: (bool) Optional. If True, saves the plot to output_file. If False, displays the plot.

    This function reads instances for each day from the CSV file, and plots these instances and their users over time.
    The x-axis represents the date, and the y-axis represents the users. Each line on the plot represents one instance.
    Depending on the value of save_plot, the plot is either saved as an image file at the specified output path or displayed.

    Note: This function assumes that the CSV file should have columns 'date', 'name', 'users'.
    """
    # Read data from csv.
    df = pd.read_csv(csv_file_path)
    df['date'] = pd.to_datetime(df['date'])

    # Pivot the dataframe for plotting.
    df_pivot = df.pivot(index='date', columns='name', values='users')

    # Create plot.
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each instance as a separate line with a unique ID.
    for i, column in enumerate(df_pivot.columns):
        df_pivot[column].plot(ax=ax, label=f"{i+1}: {column}")

        # Add a reference number at the start of each line.
        start_date = df_pivot[column].first_valid_index()
        start_value = df_pivot.loc[start_date, column]
        ax.text(start_date, start_value, str(i+1), verticalalignment='bottom', horizontalalignment='right')

    # Customize plot appearance.
    plt.xlabel('Date')
    plt.ylabel('Users')
    plt.title('Instances Over Time')
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(plt.FixedFormatter(df_pivot.index.to_series().dt.strftime("%Y-%m-%d")))

    # Format y-axis numbers to show in K (thousands).
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/1000), ',') + 'K'))

    # Move the legend outside of the plot.
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Save the plot as an image or display it.
    if save_plot:
        plt.savefig(output_file, bbox_inches='tight')
    else:
        plt.show()
