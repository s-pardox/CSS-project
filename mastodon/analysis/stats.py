import pandas as pd
import numpy as np

from mastodon.utils import data_mng

def RQ2(filepath: str) -> None:
    df = data_mng.load_from_csv_to_df(filepath)
    calculate_user_statistics(df)

def calculate_user_statistics(df: pd.DataFrame) -> None:
    """
    This function takes a pandas DataFrame as input, which should contain data about user engagement on a platform.
    It calculates and prints various statistics, such as the correlation between 'incremental_new_users' and 'toots_on_enrollment_day',
    the ratio of 'toots_on_enrollment_day' to 'incremental_new_users' (engagement_ratio), and the correlation of 'incremental_new_users'
    with the engagement in the next 3 and 7 days.

    The DataFrame should have the following columns:
    'enrollment_day': Date of user enrollment. Expected to be of datetime type.
    'incremental_new_users': Day-to-day growth of the users on the platform.
    'toots_on_enrollment_day': The number of unique users who made a toot on the same day they created their account.
    'toots_in_next_3_days': The number of unique users who made a toot in the three days following
        their account creation, not counting any toots made on the enrollment day.
    'toots_days_4_to_7': The count of distinct new users who made toots from the fourth to the seventh
        day after their enrollment, excluding the enrollment day and the three days following.

    :param df: DataFrame containing the data.
    :return: None
    """

    correlation = df['incremental_new_users'].corr(df['toots_on_enrollment_day'])
    print(f"The correlation between incremental_new_users and toots_on_enrollment_day is {correlation}")

    # Create a new column 'engagement_ratio'. If 'incremental_new_users' is 0, then the result is NaN, otherwise it's
    # 'toots_on_enrollment_day' / 'incremental_new_users'.
    df['engagement_ratio'] = np.where(df['incremental_new_users'] == 0, np.nan,
                                      df['toots_on_enrollment_day'] / df['incremental_new_users'])
    # Calculate the average, skipping NaN values
    average_engagement_ratio = df['engagement_ratio'].mean(skipna=True)
    print(f"Average engagement ratio: {average_engagement_ratio}")

    correlation_3_days = df['incremental_new_users'].corr(df['toots_in_next_3_days'])
    correlation_7_days = df['incremental_new_users'].corr(df['toots_days_4_to_7'])

    print(f"The correlation with incremental_new_users is {correlation_3_days} for the next 3 days after enrollment, and {correlation_7_days} for days 4-7 after enrollment.")

def set_pandas_options() -> None:
    """
    Sets specific display options for pandas DataFrame.
    """
    pd.set_option('display.max_rows', 5000)
    pd.set_option('display.max_columns', 5000)
    pd.set_option('display.width', 10000)
    pd.set_option('display.max_colwidth', 1000)

def calc_en_posts(df: pd.DataFrame) -> None:
    """
    This function calculates and prints the percentage of non-English posts in the input DataFrame.

    The DataFrame should have a 'lang' column, indicating the language of each post. This function first groups the DataFrame
    by 'lang', counts the number of posts in each language, and then calculates the percentage of posts that are not in English.

    :param df: DataFrame containing the data, expected to have a 'lang' column.
    :return: None
    """
    df_freq = df.groupby(pd.Grouper(key='lang')).size().reset_index(name='count')

    total_posts = len(df)
    non_en_posts = df_freq.loc[df_freq['lang'] != 'en', 'count'].sum()
    perc = round(non_en_posts * 100 / total_posts, 2)
    print(f"Total posts: {total_posts}")
    print(f"Non english posts: {non_en_posts}, {perc}%")

def calc_posts_per_instance(df: pd.DataFrame) -> None:
    """
    This function calculates and prints the number of posts per instance in the input DataFrame, in descending order.

    The DataFrame should have an 'instance' column, indicating the instance of each post. This function first groups the
    DataFrame by 'instance', counts the number of posts per instance, and then sorts the result in descending order.

    :param df: DataFrame containing the data, expected to have an 'instance' column.
    :return: None
    """
    df_freq = df.groupby(pd.Grouper(key='instance')).size().reset_index(name='count').sort_values(by=['count'],
                                                                                                  ascending=False)
    print(df_freq)

def calculate_instance_average_monthly_increment(csv_file_path: str, instance_name: str, start_date: str,
                                                 end_date: str) -> tuple:
    """
    Calculate the average monthly users increment and the average monthly percentage increment for a specific instance.

    :param csv_file_path: (str) File path to read the CSV data.
    :param instance_name: (str) Name of the instance for which to calculate the increments.
    :param start_date: (str) The start date in 'YYYY-MM-DD' format.
    :param end_date: (str) The end date in 'YYYY-MM-DD' format.

    :return: (tuple) A tuple containing the average monthly users increment and the average monthly percentage increment for the specified instance.

    This function reads instances for each day from the CSV file, selects the data for the specified instance,
    and calculates the average monthly users increment and the average monthly percentage increment.
    The calculations are performed over the period specified by the start and end dates.

    Note: This function assumes that the CSV file should have columns 'date', 'name', 'users'.
    """
    # Read data from csv.
    df = pd.read_csv(csv_file_path)

    # Convert date column to datetime and set it as index.
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Filter the dataframe for the specific instance and date range.
    df_instance = df.loc[start_date:end_date]
    df_instance = df_instance[df_instance['name'] == instance_name]

    # Resample to monthly frequency, calculate the mean of users.
    monthly_users = df_instance['users'].resample('M').mean()

    # Compute the increment and percentage increment between consecutive months.
    monthly_increment = monthly_users.diff()
    monthly_percentage_increment = monthly_users.pct_change() * 100

    # Calculate the average increment and average percentage increment.
    avg_monthly_increment = monthly_increment.mean()
    avg_monthly_percentage_increment = monthly_percentage_increment.mean()

    # Print the results.
    print(
        f"The average monthly users increment for '{instance_name}' between {start_date} and {end_date} is approximately {avg_monthly_increment:.2f} users.")
    print(
        f"The average monthly percentage increment for '{instance_name}' between {start_date} and {end_date} is approximately {avg_monthly_percentage_increment:.2f}%.")

    # Return the averages.
    return avg_monthly_increment, avg_monthly_percentage_increment

def calculate_total_average_monthly_increment(csv_file_path: str, start_date: str, end_date: str) -> float:
    """
    Calculate the average monthly users increment.

    :param csv_file_path: (str) File path to read the CSV data.
    :param start_date: (str) The start date in 'YYYY-MM-DD' format.
    :param end_date: (str) The end date in 'YYYY-MM-DD' format.

    :return: (float) The average monthly users increment.

    This function reads the CSV file, selects the data within the specified date range,
    and calculates the average monthly users increment.
    The calculation is based on the 'total_accounts' column and performed over the period specified by the start and end dates.

    Note: This function assumes that the CSV file should have columns 'created_at' and 'total_accounts'.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Convert 'created_at' to datetime and set as index
    df['created_at'] = pd.to_datetime(df['created_at'])
    df.set_index('created_at', inplace=True)

    # Filter data between start_date and end_date
    df = df.loc[start_date:end_date]

    # Resample to get the 'total_accounts' at the end of each month
    monthly_totals = df['total_accounts'].resample('M').last()

    # Calculate the difference between each month to get the increment
    monthly_increments = monthly_totals.diff()

    # Calculate the average increment
    avg_monthly_increment = monthly_increments.mean()

    # Pretty print the average monthly increment
    print(f'The average monthly increment of users from {start_date} to {end_date} is {avg_monthly_increment:.2f}.')

    return avg_monthly_increment
