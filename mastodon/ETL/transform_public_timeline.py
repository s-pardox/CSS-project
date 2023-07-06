import json
import os
from typing import List, Dict
from tqdm import tqdm
import time
from datetime import datetime, timezone
from dateutil.parser import parse

from mastodon.utils.Database import Database

def load_json_file(file_path: str) -> List[Dict]:
    """
    Load the JSON file from the given path.

    :param file_path: str, path to the JSON file
    :return: List[Dict], the loaded data
    """
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def insert_into_db(toot: Dict) -> None:
    """
    Insert the given toot data into the database.

    :param toot: Dict, the toot data to insert
    :return: None
    """
    cnx = Database.getInstance().get_connection()
    # Insert toot details and user details into the database.
    account = toot['account']
    cursor = cnx.cursor()

    # Check if the account already exists.
    check_account_exists = "SELECT * FROM Accounts WHERE account_id = %s"
    cursor.execute(check_account_exists, (account['id'],))
    existing_account = cursor.fetchone()

    # Now 'username' contains the username and 'instance' contains the instance.
    # If there was no '@' in the 'acct' string, 'instance' will be 'mastodon.social'
    username, instance = account['acct'].split('@') if '@' in account['acct'] else \
        (account['acct'], 'mastodon.social')

    # Check if the creation date of the account is not '1970-01-01' (or lower), which is often used as a placeholder for
    # missing or null dates.
    try:
        created_at_date = datetime.fromisoformat(account['created_at'].replace("Z", "+00:00"))
        if created_at_date.year <= 1970:
            raise ValueError
    except ValueError:
        # If the year is 1970 or in case of another ValueError, use a placeholder date '2000-01-01'
        created_at_date = datetime(2000, 1, 1, tzinfo=timezone.utc)

    # If the account does not exist, insert into Accounts table.
    if not existing_account:
        add_account = ("INSERT INTO Accounts "
                       "(account_id, username, instance, created_at) "
                       "VALUES (%s, %s, %s, %s)")

        data_account = (account['id'], account['username'], instance, created_at_date)
        cursor.execute(add_account, data_account)

    # Insert into AccountStats table.
    add_account_stats = ("INSERT INTO AccountStats "
                         "(account_id, followers_count, following_count, statuses_count, last_status_at) "
                         "VALUES (%s, %s, %s, %s, %s)")
    data_account_stats = (account['id'], account['followers_count'],
                          account['following_count'], account['statuses_count'], account['last_status_at'])
    cursor.execute(add_account_stats, data_account_stats)

    # Check if the creation date of the account is not '1970-01-01' (or lower), which is often used as a placeholder for
    # missing or null dates.
    try:
        created_at_date = datetime.fromisoformat(toot['created_at'].replace("Z", "+00:00"))
        if created_at_date.year <= 1970:
            raise ValueError
    except ValueError:
        # If the year is 1970 or in case of another ValueError, use a placeholder date '2000-01-01'
        created_at_date = datetime(2000, 1, 1, tzinfo=timezone.utc)

    # Check if the toot already exists.
    check_toot_exists = "SELECT * FROM Toots WHERE toot_id = %s"
    cursor.execute(check_toot_exists, (toot['id'],))
    existing_toot = cursor.fetchone()

    # If the toot does not exist, insert into Toots table.
    if not existing_toot:
        add_toot = ("INSERT INTO Toots "
                    "(toot_id, created_at, account_id) "
                    "VALUES (%s, %s, %s)")
        data_toot = (toot['id'], created_at_date, account['id'])
        cursor.execute(add_toot, data_toot)

    # Make sure data is committed to the database.
    cnx.commit()
    cursor.close()

def load_and_insert_files(directory_path: str) -> None:
    """
    Load JSON files from the specified directory and insert data into a MySQL database.

    :param directory_path: str, the path to the directory where JSON files are located.
    :return: None
    """
    # Get a sorted list of all JSON files in the directory
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])

    total_files = len(json_files)
    processed_files = 0

    # For tracking time taken.
    start_time = time.time()
    total_time = 0

    for file_name in tqdm(json_files, desc='Processing files'):
        file_start_time = time.time()

        # Full path to the file.
        file_path = os.path.join(directory_path, file_name)

        print(f"Processing file: {file_name}")

        toots = load_json_file(file_path)

        # Insert data into the database.
        for toot in toots:
            insert_into_db(toot)

        processed_files += 1
        file_end_time = time.time() - file_start_time
        total_time += file_end_time

        # Estimated remaining time.
        avg_time_per_file = total_time / processed_files
        remaining_files = total_files - processed_files
        estimated_remaining_time = avg_time_per_file * remaining_files

        # Round the estimated remaining time to the nearest whole number.
        estimated_remaining_time_rounded = round(estimated_remaining_time)

        print(f"\nEstimated remaining time: {estimated_remaining_time_rounded} seconds.")

def do(directory_path: str):
    """
    This function serves as the main entry point of the script.
    It connects to the database, loads and inserts data from all JSON files in the specified directory,
    and finally closes the database connection.

    :param directory_path: str, the path to the directory where JSON files are located
    :return: None
    """
    # Load and insert files.
    load_and_insert_files(directory_path)

    # Close the connection
    Database.getInstance().close_connection()
