import os
import json
import re
from datetime import datetime, timezone
from typing import List

from mastodon.utils import data_mng
from mastodon.utils.Database import Database

def extract_instance_info(dir_path: str, instance_name: str = None) -> List[dict]:
    """
    Extract information of a specific instance from all json files in a directory.

    :param dir_path: str, the path of the directory containing the json files
    :param instance_name: str, optional, the name of the instance to extract information from
    :return: List[dict], a list of dictionaries, each containing information of the instance from a file
    """
    instance_info = []

    # Sort files to process them in order.
    files = sorted(os.listdir(dir_path))

    print("Starting to process JSON files...")
    for filename in files:
        if filename.endswith(".json"):
            print(f"Processing {filename}...")
            with open(os.path.join(dir_path, filename), 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    print(f'Error decoding JSON from file {filename}')
                    continue

                for instance in data:
                    # Only process instances that match instance_name if it's provided
                    if instance_name is None or instance['name'] == instance_name:
                        # Extract date and time components using regular expressions
                        date_match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})Z', filename)
                        date = date_match.group(1)
                        hour = int(date_match.group(2))
                        minute = int(date_match.group(3))
                        second = int(date_match.group(4))

                        # Create a valid datetime object with UTC timezone
                        created_at = datetime(int(date[:4]), int(date[5:7]), int(date[8:10]), hour, minute, second,
                                              tzinfo=timezone.utc)

                        # Format datetime object as string in ISO 8601 format.
                        instance['created_at'] = created_at.strftime('%Y-%m-%dT%H:%M:%S.000Z')
                        instance_info.append(instance)

    print("Finished processing all JSON files.")
    return instance_info

def insert_instances_into_DB(filepath: str) -> None:
    """
    Extract information of all instances from a json file and insert them into the database.

    :param filepath: str, the path of the json file containing the instances data
    """
    cnx = Database.getInstance().get_connection()
    cursor = cnx.cursor()

    print("Starting to process the JSON file...")

    # SQL INSERT statement.
    query = """
        INSERT INTO InstanceInfo (name, users, statuses, connections, created_at)
        VALUES (%s, %s, %s, %s, %s);
    """

    total_instances = 0
    for data in data_mng.load_from_json_in_chunks(filepath):
        for instance in data:
            # Convert the ISO 8601 datetime string to MySQL's datetime format.
            created_at = datetime.strptime(instance['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')
            created_at = created_at.strftime('%Y-%m-%d %H:%M:%S')

            # Execute the query with the given instance information.
            cursor.execute(query, (instance['name'], instance['users'], instance['statuses'],
                                   instance['connections'], created_at))
            total_instances += 1

            if total_instances % 1000 == 0:
                print(f"Inserted {total_instances} instances into the database...")

    cnx.commit()
    print(f"Finished inserting {total_instances} instances into the database.")

    cursor.close()
    Database.getInstance().close_connection()

def do(dir_path: str, output_filename: str, instance_name: str = None) -> None:
    """
    Extract information of a specific instance from all json files in a directory and write to a json file.

    :param dir_path: str, the path of the directory containing the json files
    :param output_filename: str, the name of the output file
    :param instance_name: str, optional, the name of the instance to extract information from
    """
    print("Starting data extraction...")
    instance_info = extract_instance_info(dir_path, instance_name)

    print("Finished data extraction. Starting data writing...")
    output_filepath = os.path.join(dir_path, output_filename)

    print(f"Writing data to {output_filename}...")
    data_mng.save_list_to_json(instance_info, output_filepath)
    print(f"Finished writing to {output_filename}.")

    print("Finished all operations.")