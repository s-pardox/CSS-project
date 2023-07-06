import pandas as pd
from collections import Counter
from typing import List, Tuple

from mastodon.utils import data_mng

def extract_most_frequents(filepath: str, start_date: str, end_date: str, threshold: int) -> list:
    """
    Analyze the hashtags in a DataFrame loaded from a given filepath. Remove specific hashtags ('twittermigration',
    'mastodonmigration'), convert the rest to lower case, and count the occurrences of each. This function will also
    filter the data based on a specified date range.

    :param filepath: str, The path to the CSV file containing the data.
    :param start_date: str, Start date for time period selection in 'YYYY-MM-DD' format.
    :param end_date: str, End date for time period selection in 'YYYY-MM-DD' format.
    :param threshold: int, The minimum frequency a hashtag must have to be included in the results.
    :return: list of tuples. Each tuple contains a hashtag and its count.
    """

    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Filter data based on date range.
    df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

    # Extract hashtags from the 'tags' column, convert to lower case, and remove specific ones.
    hashtags = df['tags'].apply(lambda x: x.lower().split(','))
    hashtags = hashtags.apply(
        lambda x: [tag.strip() for tag in x if tag.strip() not in ['twittermigration', 'mastodonmigration']])

    # Get a list of all hashtags.
    all_hashtags = [tag for tags in hashtags for tag in tags]

    # Get a count of each hashtag.
    hashtag_counts = Counter(all_hashtags)

    # Filter hashtag_counts based on threshold.
    filtered_hashtag_counts = {tag: count for tag, count in hashtag_counts.items() if count >= threshold}

    # Sort the hashtags by their counts.
    sorted_hashtag_counts = sorted(filtered_hashtag_counts.items(), key=lambda x: x[1], reverse=True)

    # Print the hashtags and their counts.
    for tag, count in sorted_hashtag_counts:
        print(f"Hashtag: {tag}, Count: {count}")

    return sorted_hashtag_counts

def correlate_topics_hashtags(topics: List[List[str]], df: pd.DataFrame, hashtags: List[Tuple[str, int]]) -> \
        List[List[Tuple[str, int]]]:
    """
    Find the most common hashtags in tweets containing the words from each topic.

    :param topics: list of lists. Each list contains the top 10 words of a topic.
    :param df: DataFrame. The DataFrame used in the LDA analysis.
    :param hashtags: list of tuples. Each tuple contains a hashtag and its count.
    :return: list of lists. Each list contains the hashtags most commonly found in tweets containing
        words from a topic.
    """
    results = []

    # Turn the list of tuples into a dict for easier lookup.
    hashtag_dict = dict(hashtags)

    for topic in topics:
        topic_hashtags = Counter()

        for word in topic:
            # Select rows where the tweet contains the current word.
            relevant_tweets = df[df['tokenized_content'].apply(lambda x: word in x)]
            # Count occurrences of each hashtag in these tweets.
            for tweet in relevant_tweets['tags']:
                for hashtag in tweet:
                    if hashtag in hashtag_dict:
                        topic_hashtags[hashtag] += 1

        # Find the 10 most common hashtags for the current topic.
        most_common_hashtags = topic_hashtags.most_common(10)
        # just append the most common hashtags.
        results.append(most_common_hashtags)

        # Print the results.
        print(f"Most frequent hashtags: {most_common_hashtags}")
        print()

    return results
