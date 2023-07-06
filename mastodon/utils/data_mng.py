import os
import pandas as pd
import json
import re
import csv
from typing import Generator
import ijson
import itertools

from typing import Optional
from typing import List, Any

from mastodon.settings import config

def save_df_to_csv(df: pd.DataFrame, path: str) -> None:
    """
    Saves the dataframe to a CSV file at the specified path.

    :param df: The dataframe to be saved.
    :param path: The path where the CSV file will be saved.
    """
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        print(f'An error occurred while saving the file: {e}')

def save_df_to_json(df: pd.DataFrame, path: str, mod: str='w') -> None:
    """
     Saves the dataframe to a JSON file at the specified path.

     :param df: The dataframe to be saved.
     :param path: The path where the JSON file will be saved.
     :param mod: The mode in which the file is opened.
     """
    try:
        with open(path, mod) as f:
            json.dump(df.to_dict(orient='records'), f, indent=4)
    except Exception as e:
        print(f'An error occurred while saving the file: {e}')

def save_dict_to_json(data: dict, path: str, mod: str='w') -> None:
    """
    Saves the dictionary to a JSON file at the specified path.

    :param data: The dictionary to be saved.
    :param path: The path where the JSON file will be saved.
    :param mod: The mode in which the file is opened.
    """
    try:
        with open(path, mod) as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f'An error occurred while saving the file: {e}')

def save_list_to_json(data: List[Any], path: str, mod: str='w') -> None:
    """
    Saves the list to a JSON file at the specified path.

    :param data: The list to be saved.
    :param path: The path where the JSON file will be saved.
    :param mod: The mode in which the file is opened.
    """
    try:
        with open(path, mod) as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f'An error occurred while saving the file: {e}')

def save_str_to_txt(text: str, path: str, mod: str='w') -> None:
    """
    Saves the string to a text file at the specified path.

    :param text: The string to be saved.
    :param path: The path where the text file will be saved.
    :param mod: The mode in which the file is opened.
    """
    try:
        with open(path, mod) as f:
            f.write(text)
    except Exception as e:
        print(f'An error occurred while saving the file: {e}')

def load_from_json_to_dict(path: str) -> Optional[dict]:
    """
    Loads a JSON file and returns its content as a dictionary.

    :param path: The path of the JSON file to be loaded.
    :return: The content of the JSON file as a dictionary, or None if the file does not exist or contains invalid JSON.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f'File {path} not found.')
        return None
    except json.JSONDecodeError:
        print(f'File {path} contains invalid JSON.')
        return None
    return data

def load_from_txt_to_str(path: str) -> Optional[str]:
    """
    Loads a text file and returns its content as a string.

    :param path: The path of the text file to be loaded.
    :return: The content of the text file as a string, or None if the file does not exist.
    """
    try:
        with open(path) as f:
            data = f.read()
    except FileNotFoundError:
        print(f'File {path} not found.')
        return None
    return data

import pandas as pd
from typing import Optional

def load_from_csv_to_df(path: str, index_col: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Loads a CSV file and returns its content as a pandas DataFrame.

    :param path: The path of the CSV file to be loaded.
    :param index_col: The column to be used as the index of the DataFrame (optional).
    :return: The content of the CSV file as a DataFrame, or None if the file does not exist, is empty, or if an error
    occurs while loading.
    """
    try:
        if index_col is None:
            df = pd.read_csv(path)
        elif index_col != 0:
            df = pd.read_csv(path, index_col=index_col)
        else:
            df = pd.read_csv(path, index_col=0)
    except FileNotFoundError:
        print(f'File {path} not found.')
        return None
    except pd.errors.EmptyDataError:
        print(f'File {path} is empty.')
        return None
    except Exception as e:
        print(f'An error occurred while loading the file: {e}')
        return None
    return df


def merge_json_files(input_directory: str, output_directory: str, output_filename: str) -> None:
    """
    Merge multiple JSON files into a single JSON file.

    :param input_directory: str, the directory where the JSON files are located
    :param output_directory: str, the directory where the output file will be saved
    :param output_filename: str, the name of the output file
    :return: None
    """
    json_data = []

    # List all files in the directory
    files = os.listdir(input_directory)
    # Sort the list in ascending order
    files.sort()

    # Iterate over all files in the directory.
    for filename in files:
        if filename.endswith('.json'):
            with open(os.path.join(input_directory, filename), 'r') as f:
                data = json.load(f)
                json_data.extend(data)

    # Create the output directory if it doesn't exist.
    os.makedirs(output_directory, exist_ok=True)

    # Create the complete output file path.
    output_file_path = os.path.join(output_directory, output_filename)

    # Save combined data to the output file.
    with open(output_file_path, 'w') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

def fix_json_sequence(json_file_path: str):
    """
    Fixes a common formatting error in a JSON file where separate JSON arrays aren't properly connected.
    The function identifies sequences that represent the end of one array and the start of another,
    specifically a closing curly brace '}', followed by a closing square bracket ']',
    followed by an opening square bracket '[', and then an opening curly brace '{',
    each potentially separated by varying amounts of whitespace.

    It replaces these sequences with a correctly formatted sequence: a closing curly brace '}',
    followed by a comma ',', and then an opening curly brace '{',
    effectively merging the separate arrays into one.

    The function operates directly on the file, replacing all occurrences of the erroneous sequence.

    :param json_file_path: Path to the JSON file that needs to be corrected.
    """
    # Erroneous sequence pattern, with allowance for varying spaces.
    original_seq_pattern = r"\}\s*\]\s*\[\s*\{"

    # Corrected sequence
    correct_seq = "},\n  {"

    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = f.read()

    # Replace erroneous sequence using regex
    corrected_data = re.sub(original_seq_pattern, correct_seq, data)

    # Write corrected JSON back to file
    with open(json_file_path, 'w') as f:
        f.write(corrected_data)

def load_from_json_to_df(path: str) -> Optional[pd.DataFrame]:
    """
    Loads a JSON file and returns its content as a pandas DataFrame.

    :param path: The path of the JSON file to be loaded.
    :return: The content of the JSON file as a DataFrame, or None if the file does not exist, is empty, or if an error
    occurs while loading.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    except FileNotFoundError:
        print(f'File {path} not found.')
        return None
    except json.JSONDecodeError:
        print(f'File {path} is not a valid JSON file.')
        return None
    except Exception as e:
        print(f'An error occurred while loading the file: {e}')
        return None

    return df

def merge_hashtag_csv(files: List[str], output_file: str) -> None:
    """
    Main function to merge multiple CSV files and sort them by 'created_at'.

    :param files: A list of paths to the CSV files to be merged.
    :param output_file: The name of the output file (will be saved in the same path as the first file in the list)
    :return: None
    """
    dfs = []  # Initialize an empty list to store dataframes

    for file in files:
        df = pd.read_csv(file)

        # Extract the hashtag from the filename and add a new column with it
        hashtag = os.path.basename(file).split('_')[0]
        df['hashtag'] = hashtag

        # Append the dataframe to the list
        dfs.append(df)

    # Concatenate the dataframes
    df = pd.concat(dfs)

    # Convert the 'created_at' column to datetime
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Sort by 'created_at'
    df = df.sort_values(by='created_at')

    # Save the sorted dataframe to a new CSV file in the same directory as the first file in the list
    output_path = os.path.join(os.path.dirname(files[0]), output_file)
    df.to_csv(output_path, index=False)

    return None

def sql_to_csv(input_file, output_file):
    """
    This function converts an SQL query output into a structured CSV format.

    The SQL output is expected to be formatted in a tabular form with '|' as column separators, as typically returned ù
    by command-line SQL clients.
    The first row should contain column names, and a row of '-' characters typically separates the column names from
    the data. Each subsequent row represents a data entry.

    :param input_file: The full path of the input text file containing the SQL output
    :param output_file: The full path where the output CSV file will be saved
    :return: None
    """
    with open(input_file, 'r') as f_input, open(output_file, 'w', newline='') as f_output:
        csv_writer = csv.writer(f_output)

        lines = f_input.readlines()
        # Extracting column names.
        header = [column.strip() for column in lines[1].strip().split('|')[1:-1]]
        csv_writer.writerow(header)

        # Start reading data from the 3rd line (index 2) onwards.
        for line in lines[2:]:
            if line.strip() and not line.startswith('+'):
                row = [element.strip() for element in
                       # Removing leading/trailing pipe characters.
                       line.strip().split('|')[1:-1]]
                csv_writer.writerow(row)

def merge_csv_files(file1_path: str, file2_path: str, output_path: str) -> None:
    """
    Merge two CSV files and write the result to a new CSV file.

    The function reads the contents of `file1_path` and `file2_path` as CSV files into pandas DataFrames.
    It then merges the DataFrames using the common columns and drops any duplicate columns with the same name.
    The merged DataFrame is written to the specified `output_path` as a new CSV file.

    :param file1_path: The path of the first CSV file to merge.
    :param file2_path: The path of the second CSV file to merge.
    :param output_path: The path where the merged CSV file will be saved.
    :return: None
    """
    # Read the CSV files into pandas DataFrames.
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Merge the DataFrames using the common columns.
    merged_df = pd.merge(df1, df2, how='outer')

    # Drop duplicate columns with the same name.
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # Write the merged DataFrame to a new CSV file.
    merged_df.to_csv(output_path, index=False)

def load_from_json_in_chunks(filepath: str, chunk_size: int = 10000) -> Generator[dict, None, None]:
    """
    Load data from a json file in chunks.

    :param filepath: str, the path of the json file
    :param chunk_size: int, the number of records to load per chunk (default is 10000)
    :return: Generator, yielding chunks of data as dictionaries
    """
    with open(filepath, 'rb') as file:  # note 'rb' for read binary
        objects = ijson.items(file, 'item')
        chunks = iter(lambda: list(itertools.islice(objects, chunk_size)), [])
        for chunk in chunks:
            yield chunk