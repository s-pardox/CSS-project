import json
import pandas as pd
import html
import re
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import numpy as np
from langdetect import detect

import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from mastodon.settings import config
from mastodon.utils import data_mng

def set_pandas_options() -> None:
    """
    Sets specific display options for pandas DataFrame.
    """
    pd.set_option('display.max_rows', 5000)
    pd.set_option('display.max_columns', 5000)
    pd.set_option('display.width', 10000)
    pd.set_option('display.max_colwidth', 1000)

    pd.set_option('float_format', '{:f}'.format)

def do(hashtag: str) -> None:
    """
    Main function to load, parse, prepare, extrapolate, and save data.

    :return: None.
    """
    set_pandas_options()

    src_filename = f'{config.DATA_FOLDER}{hashtag}_toots.json'
    # In the case the JSON isn't in a regular format decomment the following line of code.
    ### data_mng.fix_json_sequence(src_filename)
    json_data = data_mng.load_from_json_to_dict(src_filename)

    df = extract_useful_fields(json_data)
    df = extract_lang(df)

    df['cleaned_content'], _, _, _ = zip(*df['content'].apply(lambda x: extract_and_remove_entities_from_toot(x)))
    # Apply the clean_and_remove_html_tags function to the cleaned_content column.
    df['cleaned_content'] = df['cleaned_content'].apply(clean_and_remove_html_tags)

    # Filter out all non-English toots.
    df = df[df['lang'] == 'en']

    df = remove_stopwords(df)

    # Apply the filter_terms_based_on_tfidf function to the DataFrame 'df' and store the result in 'filtered_df',
    # which contains an additional 'filtered_content' column with the filtered terms based on TF-IDF thresholds.
    df = filter_terms_based_on_tfidf(df)

    # Remove all rows with empty or very short cleaned toots.
    df = df[(df['filtered_content'] != '') & (df['filtered_content'].str.len() >= 4)]

    df = add_data_field(df)

    df['id'] = df['id'].astype(str)
    dst_filename = f'{config.DATA_FOLDER}{hashtag}_toots.csv'

    data_mng.save_df_to_csv(df, dst_filename)

def extract_useful_fields(j_data: dict) -> pd.DataFrame:
    """
    This function takes in raw JSON data from the Mastodon API and converts it into a cleaned-up pandas DataFrame.

    Specifically, it performs the following operations:
        - Selects specific fields from the raw data to be included in the DataFrame.
        - Splits the 'account' field into user and instance and keeps only the instance.
        - Converts the list of dictionaries in 'tags' and 'mentions' fields into comma-separated strings.
        - Unescapes HTML characters in the 'content' field.

    Note: This function currently defaults to 'mastodon.social' for any 'account' field that does not contain '@'.
          Future versions should replace this with the queried instance.

    :param j_data: A dictionary containing raw data from the Mastodon API.
    :return: A DataFrame containing the cleaned-up data.
    """

    # Select the columns we are interested to transfer to the dataframe
    df_data = [(
        d['id'], d['created_at'], d['queried_instance'], d['language'], d['content'], d['account']['acct'], d['tags'],
        d['mentions']
    ) for d in j_data]
    # Create the dataframe
    df = pd.DataFrame(df_data, columns=['id', 'created_at', 'queried_instance', 'lang', 'content', 'acct', 'tags',
                                        'mentions'])

    # We've interrogated a random instance (from a list of few) for the federated timeline: we're interested to keep
    # the instance name in which the toot has been published.
    # E.g.: from 'biyolokum@mstdn.science' to 'mstdn.science'
    df[['username', 'instance']] = df['acct'].str.split('@', expand=True)
    # Fill 'instance' field with 'queried_instance' value if 'instance' is empty.
    df['instance'] = df.apply(lambda row: row['queried_instance'] if pd.isnull(row['instance']) else row['instance'],
                              axis=1)

    """
    From a list of dictionaries:
        "tags": [
            {
                "name": "twittermigration",
                "url": "https://mastodon.social/tags/twittermigration"
            },
            ...
        ]
    to a simpler comma separated list of joined elements: 
        twittermigration, ...
    """
    df['tags'] = df['tags'].apply(lambda x: ', '.join([i['name'] for i in x]))

    """
    The same for mentions.
        "mentions": [
            {
                "id": "13179",
                "username": "Mastodon",
                "url": "https://mastodon.social/@Mastodon",
                "acct": "Mastodon"
            },
            ...
        ]
    """
    df['mentions'] = df['mentions'].apply(
        lambda x: ', '.join(
            [re.sub('^(?:http[s]?://)?(?:www\\.)?([^/]+)(/|$)', '\\1', i['url']) for i in x]
        )
    )

    """
    df['tags'] = df['tags'].apply(lambda x: [i.lower() for i in x])
    df['mentions'] = df['mentions'].apply(lambda x: [i.lower() for i in x])
    """

    # Unescape the escaped characters
    df['content'] = df['content'].apply(html.unescape)

    return df

def extract_and_remove_entities_from_toot(orig_toot: str) -> tuple[str, list, list, list]:
    """
    This function processes a single toot by extracting and returning hashtags, URLs, and mentions, while simultaneously
    removing these entities from the original toot. This facilitates cleaner text data for further analysis and
    preserves extracted information for additional research purposes.

    :param orig_toot: The original toot text as a string.
    :return: A tuple containing the sanitized toot text and lists of extracted hashtags, URLs, and mentions.
    """

    # Parse the HTML of the toot.
    soup = BeautifulSoup(orig_toot, 'html.parser')
    # Work in progress cleaned toot.
    toot = orig_toot

    hashtags = []
    mentions = []
    links = []

    """
    Hashtag HTML formatting:
    <a href="https://mastodon.social/tags/joinmastodon" class="mention hashtag" rel="tag">
        #<span>joinmastodon</span>
    </a>
    """
    # Replace the span elements that contain hashtags with a new span element that contains the text "#hashtag"
    for elm in soup.find_all('a', {'class': 'mention hashtag'}):
        hashtag = ' ' + elm.text + ' '
        hashtags.append(hashtag)
        elm.replaceWith(hashtag)

    """
    <a class="hashtag" href="https://posts.stream/tag/askfedi" rel="nofollow noopener noreferrer" target="_blank">
        #AskFedi
    </a>
    
    What's the real difference?!
    """
    for elm in soup.find_all('a', {'class': 'hashtag'}):
        hashtag = ' ' + elm.text + ' '
        hashtags.append(hashtag)
        elm.replaceWith(hashtag)

    """
    Mention HTML formatting:
    <span class="h-card">
        <a href="https://mastodon.social/@Mastodon" class="u-url mention">
            @<span>Mastodon</span>
        </a>
    </span> 
    
    Why h-card? I suppose that it's related to the automatic generation of a "rich preview card" from an url that
    embeds all the needed tags.
    https://docs.joinmastodon.org/entities/PreviewCard/   
    """
    # Same for @mentions
    for elm in soup.find_all('span', {'class': 'h-card'}):
        mention_url = ' ' + elm.next_element.attrs['href'].split('//')[1].replace('/', '') + ' '
        mentions.append(mention_url)
        elm.replaceWith('')

    """
    Have to be analyzed 'a' tags representing: 
        case 1. mentions, that haven't the usual '<span class="h-card">'
        case 2. hashtags, that haven't any 'hashtag' class
    
    <p>
        <a href="https://kafeneio.social/@foufoutos" class="u-url mention" rel="nofollow noopener noreferrer" target="_blank">
            @foufoutos@kafeneio.social
        </a>
        <span> Εύγε νέε μου! </span>🐦
        <span><br><br></span>
        <a href="https://electricrequiem.com/tags/twittermigration" rel="nofollow noopener noreferrer" target="_blank">
            #twittermigration
        </a>
        <span> <br><br></span>
        <a href="https://www.youtube.com/watch?v=ODIvONHPqpk" rel="nofollow noopener noreferrer" target="_blank">
            https://www.youtube.com/watch?v=ODIvONHPqpk
        </a>
    </p>
    """

    """
    External link usual formatting:
    <a href="https://www.washingtonpost.com/technology/2023/01/05/twitter-blue-verification/" 
        rel="nofollow noopener noreferrer" target="_blank">
        <span class="invisible">
            https://www.</span>
        <span class="ellipsis">
            washingtonpost.com/technology/
        </span>
        <span class="invisible">
            2023/01/05/twitter-blue-verification/
        </span>
    </a>
    """
    for elm in soup.find_all('a'):
        link = elm.text
        links.append(link)

        # Extracting all the next 'span' siblings.
        span_elms = elm.findAll('span')
        if len(span_elms) > 0:
            visible_url = ''
            for span_elm in span_elms:
                if 'class' in span_elm.attrs.keys():
                    visible_url += span_elm.text if 'invisible' not in span_elm.attrs['class'] else ''

            # It could be the case that the visible URL is exactly the same of the href URL
            visible_url = link if visible_url == '' else visible_url
            elm.replaceWith('')

        # case 1. (mentions)
        elif elm.text.startswith('@'):
            mentions.append(elm.text)
            elm.replaceWith('')

        # case 2. (hashtags)
        elif elm.text.startswith('#'):
            hashtags.append(elm.text)
            elm.replaceWith('')

    """
    Moreover, there could be toots in which there aren't any explicit hashtags in the text:
        <p>
            Twitter Blue Still Broken
        </p>
        <p>
            <a href="https://friendica.myportal.social/display/e65e1095-4263-b743-3946-754400142330" rel="nofollow 
                noopener noreferrer" target="_blank">
                <span class="invisible">
                    https://</span><span class="ellipsis">friendica.myportal.social/disp
                </span>
                <span class="invisible">
                    lay/e65e1095-4263-b743-3946-754400142330
                </span>
            </a>
        </p>
        
        ..but, at the same, the 'tags' data section contains hashtags!
        
        "tags": [
            {
                "name": "twitter",
                "url": "https://mastodon.social/tags/twitter"
            },
            {
                "name": "deletetwitter",
                "url": "https://mastodon.social/tags/deletetwitter"
            },
            {
                "name": "twittermigration",
                "url": "https://mastodon.social/tags/twittermigration"
            }
        ]
    """

    # Let's permanently clean the toot (removing hashtags, mentions, and URLs).
    def remove_hashtags(text):
        return re.sub(r'#\w+', '', text)
    toot = remove_hashtags(str(soup))

    return toot, hashtags, mentions, links


def clean_and_remove_html_tags(toot: str) -> str:
    """
    This function preprocesses a toot by performing the following operations:
    - Removing HTML tags
    - Removing non-word and non-whitespace characters
    - Unescaping escaped characters
    - Removing meaningless short words
    - Removing numbers
    - Replacing multiple whitespace characters with a single space
    - Trimming whitespace characters from the beginning and end of the resulting string

    This results in a cleaned toot text suitable for further text analysis tasks.

    :param toot: The original toot text as a string.
    :return: The cleaned toot text as a string.
    """
    # Remove all the remaining HTML tags using a regular expression (mostly <p></p> and <br/>).
    toot = re.sub(r'<[^>]*>', ' ', toot)

    # Add this line to remove sequences of underscores that are considered as whole words (e.g., "______"), replacing
    # them with a single space in the text.
    toot = re.sub(r'\b_+\b', ' ', toot)

    # Unescape the escaped characters.
    toot = html.unescape(toot)

    # Remove non-word and non-whitespace characters and replace them with a single space.
    toot = re.sub(r'[^\w\s]+', ' ', toot)

    # Remove meaningless short words (e.g., words with 1 or 2 characters).
    toot = re.sub(r'\b\w{1,2}\b', '', toot)

    # Remove numbers.
    toot = re.sub(r'\d+', '', toot)

    # Remove non-ASCII characters.
    toot = re.sub(r'[^\x00-\x7F]+', '', toot)

    # Replace multiple whitespace characters with a single space.
    toot = re.sub(r'\s+', ' ', toot)

    # Remove any whitespace characters from the beginning and end of the resulting string.
    toot = toot.strip()

    return toot

def extract_lang(df: pd.DataFrame) -> pd.DataFrame:
    """
     This function takes a dataframe as input and extracts the language of the toots.
     If the 'lang' column is None or empty for a given row, the function uses the 'detect' function
     from the langdetect library on the 'content' of the toot to determine the language.
     The dataframe is then returned with the 'lang' column updated.

     :param df: A DataFrame containing toots data.
     :return: The input DataFrame with the 'lang' column updated.
     """
    df['lang'] = df.apply(
        lambda row: detect(row['content']) if row['lang'] is None or len(row['lang']) == 0 else row['lang'], axis=1)

    return df

def add_data_field(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function adds a new column 'date' to the dataframe. The 'date' column is derived from the 'created_at' column,
    which is converted to datetime and then only the date part is retained. The 'date' column is inserted at the
    second position (loc=1) in the dataframe.

    :param df: A DataFrame containing toot data.
    :return: The input DataFrame with an additional 'date' column.
    """
    # Convert 'created_at' to datetime and extract the date.
    date_series = pd.to_datetime(df['created_at']).dt.date
    # Insert the date column after 'created_at'.
    df.insert(loc=1, column='date', value=date_series)

    return df

def remove_stopwords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove English stopwords from the `cleaned_content` column of a pandas DataFrame.
    This function uses both NLTK and SpaCy libraries to generate a comprehensive list of English stopwords.
    Then, it applies a custom function to the `cleaned_content` column of the input DataFrame to remove all stopwords.
    The text is first converted to lowercase to ensure consistency.

    :param df: Input pandas DataFrame. Expected to have a 'cleaned_content' column containing text data.
    :return: Modified DataFrame where the 'cleaned_content' column has been updated to remove all English stopwords.
    """
    # Load spaCy model.
    nlp = spacy.load('en_core_web_sm')

    # Get the set of stopwords from both NLTK and spaCy.
    stop_words = set(stopwords.words('english')).union(nlp.Defaults.stop_words)

    # Define a function to remove stopwords from a text.
    def stopwords_cleaner(text):
        return ' '.join([word for word in str(text).lower().split() if word not in stop_words])

    df['cleaned_content'] = df['cleaned_content'].apply(lambda x: stopwords_cleaner(x))

    return df

def filter_terms_based_on_tfidf(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function filters the terms in the 'cleaned_content' column of a DataFrame based on the TF-IDF thresholds
    (max_df and min_df). It returns the DataFrame with an additional 'filtered_content' column containing the
    filtered texts.
    """
    # Create a TfidfVectorizer instance without the stop_words parameter.
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)

    # Fit the vectorizer on the cleaned_content column to get the vocabulary.
    vectorizer.fit(df['cleaned_content'])

    # Get the terms that satisfy the max-min thresholds.
    valid_terms = set(vectorizer.vocabulary_.keys())

    def filter_terms(row_text):
        """
        Filters the tokens in a string based on the 'valid_terms' set.
        This function receives a string, splits it into tokens and keeps only those tokens that are present
        in the 'valid_terms' set (which is expected to be in the outer scope). The resulting tokens are then
        joined back into a string and returned.

        :param row_text: A string of space-separated tokens.
        :return: A string of space-separated tokens, where only tokens present in 'valid_terms' have been kept.
        """
        tokens = row_text.split()
        filtered_tokens = [token for token in tokens if token in valid_terms]
        return ' '.join(filtered_tokens)

    # Apply the filtering to the cleaned_content column.
    df['filtered_content'] = df['cleaned_content'].apply(filter_terms)

    return df

def print_dataframe_info(df: pd.DataFrame) -> None:
    """
    This function receives a pandas DataFrame as input and prints out useful information about the DataFrame.
    Specifically, it will print:
        - The number of rows in the DataFrame.
        - Whether there are any null values in the DataFrame.
        - The first 30 rows of the DataFrame to provide a preview of the data.
        - The data types of the columns in the DataFrame.

    :param df: The pandas DataFrame to print information about.
    :return: None
    """
    # Print the number of rows in the DataFrame
    num_rows = len(df)
    print(f"Number of rows: {num_rows}")

    # Check and print if there are any null values in the DataFrame
    has_null = df.isnull().values.any()
    print(f"Contains Null values: {has_null}")

    # Print a preview of the DataFrame (first 30 rows)
    preview = df.head(30)
    print(f"Data preview:\n{preview}")

    # Print the data types of the columns in the DataFrame
    column_data_types = df.dtypes
    print(f"Column data types:\n{column_data_types}")