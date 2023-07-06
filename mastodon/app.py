from datetime import datetime, timedelta
import pandas as pd

from mastodon.ETL import extract_hashtag
from mastodon.ETL import extract_user_toots
from mastodon.ETL import extract_public_timeline
from mastodon.ETL import transform_hashtag
from mastodon.ETL import transform_user_toots
from mastodon.ETL import transform_public_timeline
from mastodon.ETL import transform_instances
from mastodon.ETL import extract_instances
from mastodon.ETL import extract_instances_social

from mastodon.utils import data_mng
from mastodon.utils import sql_tables

from mastodon.analysis import plots
from mastodon.analysis import LDA
from mastodon.analysis import hashtags
from mastodon.analysis import DTM
from mastodon.analysis import DTM_plots
from mastodon.analysis import stats
from mastodon.analysis import sentiment

from mastodon.settings import config
from mastodon.settings import github_credentials
from mastodon.settings import instances_social_credentials

def launch() -> None:
    """
    This function serves as the command center for the entire data processing pipeline,
    encompassing everything from data retrieval to analysis.

    For usage, progressively uncomment the statements that correspond to the steps you
    wish to activate in the pipeline.

    Note: Each step is currently commented out to prevent unintentional execution.
    Remember to save your work before uncommenting and running each step.
    """

    ### I. ETL pipeline. ###

    ### 1. ###
    # Retrieve toots referring to the selected hashtag from a specific Mastodon instance's API,
    # without any timeframe limitation.
    """
    hashtags_to_retrieve = ['twittermigration', 'mastodonmigration']

    for hashtag in hashtags_to_retrieve:
        extract_hashtag.do(hashtag)

    files = []
    for hashtag in hashtags_to_retrieve:
        transform_hashtag.do(hashtag)
        files.append(f'{config.DATA_FOLDER}{hashtag}_toots.csv')
    data_mng.merge_hashtag_csv(files, 'merged_hashtags_toots.csv')
    """

    ### 2. ###
    # Extract all toots related to 2 specific Mastodon users who have published statistics,
    # with a specified cut-off date parameter within the script.
    """
    extract_user_toots.do()
    transform_user_toots.do()
    """

    ### 3. ###
    # Retrieve all toots published in the public timeline for a specific Mastodon instance
    # within the specified time range, as defined within the script.
    """
    extract_public_timeline.do('mastodon.social')
    """

    # Generate the needed SQL tables.
    """
    sql_tables.drop_main_tables()
    sql_tables.create_main_tables()
    """

    # Populate the database.
    """
    # transform_public_timeline.do(config.DATA_PUBLIC_BATCHES_FOLDER)
    """

    ### 4. ###
    # Extract the instances.json file from the specified GitHub repository.
    """
    extract_instances.do('simonw', 'scrape-instances-social', github_credentials.ACCESS_TOKEN, '2022-11-20', '2022-12-21',
                         config.DATA_INSTANCES_FOLDER)
    """

    # Process the raw data and generate a JSON file; creates the needed SQL table and ingest the data.
    """
    transform_instances.do(config.DATA_INSTANCES_FOLDER, 'mastodon.social',
                           'mastodon.social-instance_2022-11-20_2022-12-21.json')
    sql_tables.create_instance_info_table()
    transform_instances.insert_instances_into_DB(f'{config.DATA_INSTANCES_FOLDER}mastodon.social-instance_2022-11-20_2022-12-21.json')
    """

    # Process and load the data on a comfortable SQL table.
    """
    sql_tables.create_top_10_instance_table()
    sql_tables.create_insert_daily_top_10_procedure()
    sql_tables.call_insert_daily_top_10_procedure()
    sql_tables.delete_insert_daily_top_10_procedure()
    """

    # Convert the SQL tabular result into a more comfortable CSV file.
    """
    data_mng.sql_to_csv(f'{config.DATA_SQL_OUTPUT_FOLDER}daily_top_10.sql',
                        f'{config.DATA_SQL_OUTPUT_FOLDER}daily_top_10.csv')
    """

    # An additional way to retrieve Mastodon statistics: https://instances.social.
    # Nevertheless, it's not working as expected due to a lack of completeness.
    """
    instances = extract_instances_social.get_top_instances(instances_social_credentials.ACCESS_TOKEN, count=10)
    for instance in instances:
        print(f"Instance: {instance['name']}, Users: {instance['users']}")
    """



    ### II. Analysis. ###

    ### Instances analysis.

    """
    Sometimes we've directly interrogated the MySQL instance console. 
    In order to convert the raw output of the SQL query into a usable CSV format, we've used the following function. 
    This is just an execution example.

    data_mng.sql_to_csv(f'{config.DATA_SQL_OUTPUT_FOLDER}daily_top_10.sql',
                        f'{config.DATA_SQL_OUTPUT_FOLDER}daily_top_10.csv')

    [...]
    """

    # Useful plots and statistics about the most significant/populated Mastodon instances.
    """
    plots.plot_instances_over_time_v1(f'{config.DATA_SQL_OUTPUT_FOLDER}daily_top_10.csv')
    stats.calculate_instance_average_monthly_increment(f'{config.DATA_SQL_OUTPUT_FOLDER}daily_top_10.csv', 'mastodon.social',
                                                       start_date='2022-11-20', end_date='2023-05-15')
    stats.calculate_instance_average_monthly_increment(f'{config.DATA_SQL_OUTPUT_FOLDER}daily_top_10.csv', 'mastodon.social',
                                                       start_date='2022-11-20', end_date='2022-12-19')
    stats.calculate_total_average_monthly_increment(f'{config.DATA_FOLDER}415114-471607_toots.csv', start_date='2022-11-20',
                                                    end_date='2023-05-15')
    """

    ### RQ1
    """
    plots.RQ1(f'{config.DATA_FOLDER}415114-471607_toots.csv', f'{config.DATA_FOLDER}events.json',
              f'{config.DATA_FOLDER}mastodonmigration-twittermigration_toots.csv')
    """

    ### RQ2
    """
    plots.RQ2(f'{config.DATA_SQL_OUTPUT_FOLDER}users_engagement.csv')
    stats.RQ2(f'{config.DATA_SQL_OUTPUT_FOLDER}users_engagement.csv')
    """

    ### RQ3
    ### Plot the instances who registered the higher number of toots with the selected hashtags.
    # plots.RQ3(f'{config.DATA_FOLDER}mastodonmigration-twittermigration_toots.csv')

    def analyze_periods(start_dates: list, num_topics: int):
        """
        This function analyzes periods of data from a given set of start dates and number of topics.
        For each start date, the function computes the most frequent hashtags and topics using the LDA model,
        and calculates correlation between the topics and hashtags. The results are appended to a list and
        converted to a pandas DataFrame for easy visualization.

        :param list start_dates: A list of start dates in string format 'YYYY-MM-DD' from which the analysis
        begins. For each start date, the analysis period is calculated as five days after the start date.

        :param int num_topics: The number of topics for the LDA model to compute for each period.

        :return: This function does not return a value. It prints a pandas DataFrame of the results
        for each topic for each period, showing the start date, LDA topic, the five most frequent hashtags,
        and the correlated hashtags if any.
        """
        results = []

        for start_date in start_dates:
            # Convert start_date to datetime and calculate end_date.
            start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_dt = start_date_dt + timedelta(days=5)
            end_date = end_date_dt.strftime('%Y-%m-%d')

            hashtags_result = hashtags.extract_most_frequents(
                f'{config.DATA_FOLDER}mastodonmigration-twittermigration_toots.csv',
                start_date, end_date, 10)

            LDA_result_1, LDA_result_2 = LDA.do(
                f'{config.DATA_FOLDER}mastodonmigration-twittermigration_toots.csv',
                num_topics=num_topics, start_date=start_date, end_date=end_date)

            result_3 = hashtags.correlate_topics_hashtags(LDA_result_1, LDA_result_2, hashtags_result)

            # For each topic, append an entry in the results.
            for i in range(num_topics):
                results.append((start_date, LDA_result_1[i], [h[0] for h in hashtags_result[:5]], result_3[i]))

        # Convert results to DataFrame.
        results_df = pd.DataFrame(results, columns=['date', 'LDA_topic', 'most_frequent_hashtags',
                                                    'eventual_correlated_hashtags'])

        # Print results.
        print(results_df.to_string(index=False))

    """
    # Most significant events within the time period of interest.
    events = ['2022-10-27', '2022-11-04', '2022-11-17', '2022-12-15', '2022-12-18']
    analyze_periods(events, num_topics=1)
    """



    ### III. Extra/attempts. ###

    ### Dynamic Topic Modeling.
    # stats.do()
    # DTM_plots.do()

    ### Sentiment analysis.
    # sentiment.do()