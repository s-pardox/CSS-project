import json
import numpy as np
import pandas as pd
import re
from typing import List, Dict, Optional, Union, Tuple
from bs4 import BeautifulSoup
from datetime import datetime

from mastodon.settings import config
from mastodon.utils import data_mng

DATA_FILE = config.DATA_FOLDER + '415114-471607_toots.json'
OUTPUT_DATA_FILE = config.DATA_FOLDER + '415114-471607_toots.csv'

def parse_record(record: dict) -> Optional[Dict[str, Union[int, datetime]]]:
    """
    Parse a record from the JSON data.

    :param record: dict, record to parse
    :return: dict or None, parsed data (total_accounts, last_hour, last_day, last_week, created_at)
    """
    soup = BeautifulSoup(record['content'], 'html.parser')

    stats = {'total_accounts': None, 'last_hour': None, 'last_day': None, 'last_week': None, 'created_at': None}

    for text in soup.stripped_strings:
        if 'accounts' in text:
            stats['total_accounts'] = int(re.sub(r'\D', '', text))
        elif 'in the last hour' in text:
            stats['last_hour'] = int(re.sub(r'\D', '', text))
        elif 'in the last day' in text:
            stats['last_day'] = int(re.sub(r'\D', '', text))
        elif 'in the last week' in text:
            stats['last_week'] = int(re.sub(r'\D', '', text))

    stats['created_at'] = datetime.fromisoformat(record['created_at'].replace('Z', '+00:00'))

    # Check if any of the mandatory fields could not be parsed.
    # We're assuming that 'total_accounts' and 'last_week' are always present.
    if stats['total_accounts'] is None or stats['last_week'] is None:
        return None

    return stats

def parse_data(data: dict) -> list:
    """
    Function to parse the data.

    :param data: Raw data in dictionary format.
    :return: Parsed data as list of dictionaries.
    """
    parsed_data = []
    for item in data:
        record_to_parse = {'content': item['content'], 'created_at': item['created_at']}
        parsed_record = parse_record(record_to_parse)
        if parsed_record is not None:
            parsed_data.append(parsed_record)
        else:
            print(record_to_parse)
    return parsed_data

def calculate_last_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function receives a pandas DataFrame as input, calculates the 'last_day' column based on
    the 'total_accounts' column and returns the DataFrame with the updated 'last_day' column.

    :param df: The pandas DataFrame to calculate 'last_day' for.
    :return: The updated pandas DataFrame.
    """
    # Calculate 'last_day' based on the 'total_accounts' column
    df['last_day_test'] = df['total_accounts'] - df['total_accounts'].shift(1)

    return df

def prepare_dataframe(parsed_data: list) -> pd.DataFrame:
    """
    Function to create and prepare a DataFrame from parsed data.

    :param parsed_data: Parsed data as list of dictionaries.
    :return: Prepared DataFrame.
    """
    df = pd.DataFrame(parsed_data)
    df = df.sort_values('created_at')
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['created_at'] = df['created_at'].dt.floor('S')
    df['created_at'] = df['created_at'].apply(lambda dt: dt.replace(second=0))
    # Convert minutes to "00" in 'created_at' column.
    df['created_at'] = df['created_at'].apply(lambda x: x.replace(minute=0))

    cols_to_check = ['total_accounts', 'last_hour', 'last_day', 'last_week', 'created_at']
    df = df.drop_duplicates(subset=cols_to_check, keep='first')
    return df

def find_missing_dates(df: pd.DataFrame, date_col: str = 'created_at', freq: str = 'H') -> pd.Index:
    """
    Find and return dates missing from a time series in a DataFrame.

    :param df: DataFrame containing the time series.
    :param date_col: Column in df containing the dates. Default is 'created_at'.
    :param freq: Frequency of the time series. Default is 'H' for hourly.
                 Use 'D' for daily, 'M' for monthly, etc. or custom frequency strings.

    :return: DatetimeIndex containing the missing dates.
    """
    # Check if 'created_at' is a column in the DataFrame.
    if date_col in df.columns:
        # If it's a column, then convert it to datetime.
        df[date_col] = pd.to_datetime(df[date_col])
    else:
        # If it's not a column, then it should be the index.
        # Make a copy of the DataFrame with 'created_at' as a column.
        df = df.reset_index().rename(columns={'index': date_col})

    # Create date range from min to max date in your data.
    full_date_range = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq=freq)

    # Extract dates that exist in your data.
    existing_dates = pd.Series(df[date_col].unique())

    # Find dates in the full range that are not in existing dates.
    missing_dates = full_date_range.difference(existing_dates)

    return missing_dates

def fill_missing_data(orig_df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Fill the missing dates in the DataFrame's index and the corresponding missing values in its columns.

    The function reindexes the DataFrame using a complete date range obtained from the minimum and maximum
    dates in the DataFrame's index, covering all hourly intervals in this date range.

    Missing values in the 'total_accounts' column are filled using linear interpolation, with the results
    rounded to ensure integer values.

    The 'created_at' column is filled with the corresponding index values.

    :param orig_df: pd.DataFrame, the input DataFrame. This DataFrame should be indexed by date and time.
    :param inplace: bool, if True modify the original DataFrame. Default is False.
    :return: pd.DataFrame, the DataFrame with missing dates and values filled.
    """
    if not inplace:
        df = orig_df.copy(deep=True)
    else:
        df = orig_df

    # First, ensure that the DataFrame is indexed by 'created_at'.
    df.set_index('created_at', inplace=True)

    # Create a date range that includes all hours in the range.
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')

    # Reindex the DataFrame to include any missing hours.
    df = df.reindex(date_range)

    # Interpolate the missing values in the 'total_accounts' column.
    df['total_accounts'] = df['total_accounts'].interpolate(method='linear').round()

    # Fill the 'created_at' column with the corresponding index values.
    df['created_at'] = df.index

    return df

def calculate_data_holes_ratio(df: pd.DataFrame, data_holes: List) -> Tuple[int, int, float]:
    """
    Calculate the ratio of data holes to total rows.

    :param df: The dataframe containing the data.
    :param data_holes: A list of missing datetime objects.
    :return: Total number of rows, Total number of data holes, Ratio of data holes to total rows.
    """
    total_rows = len(df)
    total_data_holes = len(data_holes)
    if total_rows != 0:
        ratio = round(total_data_holes / total_rows, 4)
    else:
        ratio = 0.0

    return total_rows, total_data_holes, ratio

def calculate_data_holes_percentage(total_rows: int, total_data_holes: int) -> float:
    """
    Calculate the percentage of data holes over the total rows.

    :param total_rows: The total number of rows.
    :param total_data_holes: The total number of data holes.
    :return: The percentage of data holes.
    """
    percentage = round((total_data_holes / total_rows) * 100, 2)

    return percentage

def write_statistics_to_file(filepath: str, data_holes: List, df: pd.DataFrame, total_rows: int,
                             total_data_holes: int, ratio: float, percentage: float) -> None:
    """
    Write the data holes statistics to a file.

    :param filepath: The output file name.
    :param data_holes: A list of missing datetime objects.
    :param df: A DataFrame containing the original data.
    :param total_rows: The total number of rows.
    :param total_data_holes: The total number of data holes.
    :param ratio: The ratio of data holes to total rows.
    :param percentage: The percentage of data holes.
    :return: None.
    """
    with open(filepath, 'w') as f:
        f.write(f"Total number of rows: {total_rows}\n")
        f.write(f"Total number of data holes: {total_data_holes}\n")
        f.write(f"Ratio of data holes to total rows: {ratio}\n")
        f.write(f"Percentage of data holes: {percentage}%\n")

        first_date = df['created_at'].min()
        last_date = df['created_at'].max()

        f.write(f"First data point date: {first_date}\n")
        f.write(f"Last data point date: {last_date}\n")

        f.write("\nData holes:\n")
        for hole in data_holes:
            f.write(f"{hole}\n")

def remove_decimal_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert individual float values in the DataFrame to int, only if they are whole numbers.

    :param df: The input DataFrame.
    :return: The DataFrame with applicable float values converted to int.
    """
    float_cols = df.select_dtypes(include=['float64']).columns

    for col in float_cols:
        df[col] = df[col].astype(str).replace('\.0$', '', regex=True).replace('nan', np.nan)

    return df

def do() -> None:
    """
    Main function to load, parse, prepare, extrapolate, and save data.

    :return: None.
    """
    # Load raw data from disk.
    data = data_mng.load_from_json_to_dict(DATA_FILE)

    parsed_data = parse_data(data)
    df = prepare_dataframe(parsed_data)

    # Let's preliminarly analyze eventual 'data holes'.
    data_holes = find_missing_dates(df)

    # Calculate the ratio and the percentage of data holes.
    total_rows, total_data_holes, ratio = calculate_data_holes_ratio(df, data_holes)
    percentage = calculate_data_holes_percentage(total_rows, total_data_holes)

    # Write the statistics to a file.
    write_statistics_to_file(f'{config.DATA_FOLDER}imputation_statistics.txt', data_holes, df, total_rows,
                             total_data_holes, ratio, percentage)

    # Fill the hour holes in the data.
    df = fill_missing_data(df, inplace=True)

    # Once we've imputed the missing data, we can now calculate the hourly increment.
    df['hourly_increment'] = df['total_accounts'].diff()
    # Calculate daily increment in 'total_accounts' column.
    df['daily_increment'] = df['total_accounts'].diff(24)

    # It will remove the current index and replace it with a default integer index.
    df = df.reset_index(drop=True)
    # Move 'created_at' column to first position.
    df = df[['created_at'] + [col for col in df.columns if col != 'created_at']]

    # Convert float columns to int.
    df = remove_decimal_zeros(df)

    # Finally, save data to disk.
    data_mng.save_df_to_csv(df, OUTPUT_DATA_FILE)