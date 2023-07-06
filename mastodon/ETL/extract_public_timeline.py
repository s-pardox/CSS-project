import requests
from datetime import datetime, timezone
from typing import Optional, Tuple
import time

from mastodon.settings import config
from mastodon.utils import data_mng
from mastodon.settings import mastodon_credentials

# Previous retrieval interval.
# LOWER_BOUND_DATE = "2022-11-27"
# UPPER_BOUND_DATE = "2022-12-04"

# Previous retrieval interval.
# LOWER_BOUND_DATE = "2022-12-05"
# UPPER_BOUND_DATE = "2022-12-19"

# Current retrieval interval.
LOWER_BOUND_DATE = "2022-12-19"
UPPER_BOUND_DATE = "2022-12-26"

# Set here your access token or use the one defined in the settings/mastodon_credentials.py.
ACCESS_TOKEN = mastodon_credentials.ACCESS_TOKEN

def save_checkpoint(instance_name: str, created_at: str, max_id: int, index: int) -> None:
    """
    Store intermediate metadata results into a JSON file.

    :param instance_name: str, instance name whose public toots we're retrieving
    :param created_at: str, the maximum creation datetime reached during the last API interaction
    :param max_id: int, the maximum id of the toots to include in the next request
    :param index: int, the current index of the saved toots
    :return: None
    """
    filename = f'{config.DATA_FOLDER}{instance_name}_checkpoint.json'
    data_mng.save_dict_to_json({'created_at': created_at, 'max_id': max_id, 'index': index}, filename)

def load_checkpoint(instance_name: str) -> Optional[Tuple[str, int, int]]:
    """
    Load any previously saved checkpoints from which retrieval process can restart.

    :param instance_name: str, instance name whose public toots you want to retrieve
    :return: Optional[Tuple[str, int, int]], a tuple containing the creation time, the maximum id of the toots to include
             in the next request and the current index of saved toots, or None if no checkpoint exists
    """
    filename = f'{config.DATA_FOLDER}{instance_name}_checkpoint.json'

    result = data_mng.load_from_json_to_dict(filename)
    if result:
        print(f"Resuming the retrieval process for instance: {instance_name}")
        print(f"Last retrieved toot created at: {result['created_at']}")
        print(f"Max id for the next request: {result['max_id']}")
        print(f"Index of the current batch: {result['index']}")
        return result['created_at'], result['max_id'], int(result['index'])
    else:
        print(f"No checkpoint found for instance: {instance_name}")
        return None

def save_results(instance_name: str, data: list, index: int) -> None:
    """
    Store batch results into a JSON file.

    :param instance_name: str, the name of the instance whose toots are being saved
    :param data: list, the toots
    :param index: int, a batch index
    """
    filename = f'{config.DATA_PUBLIC_BATCHES_FOLDER}{instance_name}_toots-{index}.json'
    data_mng.save_list_to_json(data, filename)

def fetch_and_save_toots(instance_name: str, lower_bound_date: str, upper_bound_date: str) -> None:
    """
    Retrieve and store a instance's 'toots' in a JSON file.
    This function fetches all toots of a given instance, starting from the most recent ones,
    and writes them into a JSON file.

    :param instance_name: str, instance name whose toots you want to retrieve
    :param lower_bound_date: str, the date from which we should start retrieving toots. Format should be 'YYYY-MM-DD'.
    :param upper_bound_date: str, the date until which we should retrieve toots. Format should be 'YYYY-MM-DD'.
    :return: None
    """
    # Convert the string dates to datetime objects.
    lower_bound_date = datetime.strptime(lower_bound_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    upper_bound_date = datetime.strptime(upper_bound_date, '%Y-%m-%d').replace(tzinfo=timezone.utc, hour=23, minute=59,
                                                                               second=59)

    # Buffer time to prevent hitting the rate limit by a narrow margin.
    buffer_time = 5

    # Set the base URL of the Mastodon instance.
    BASE_URL = f'https://{instance_name}'

    # Set the API endpoint for retrieving public timeline.
    TIMELINE_ENDPOINT = '/api/v1/timelines/public'

    # Are there any previously saved checkpoints from which we can restart the retrieval process?
    checkpoint = load_checkpoint(instance_name)
    if checkpoint is not None:
        created_at, max_id, index = checkpoint
    else:
        max_id = None
        # Index used as suffix in the batch filename.
        index = 0

    headers = {'Authorization': f'Bearer {ACCESS_TOKEN}'}

    while True:
        if max_id is not None:
            response = requests.get(
                f"{BASE_URL}{TIMELINE_ENDPOINT}",
                params={'max_id': max_id, 'limit': 200},
                headers=headers
            )
        else:
            response = requests.get(
                f"{BASE_URL}{TIMELINE_ENDPOINT}",
                params={'limit': 200},
                headers=headers
            )

        print('Rate limit:', response.headers.get('X-RateLimit-Limit'))
        print('Remaining:', response.headers.get('X-RateLimit-Remaining'))
        print('Reset time:', response.headers.get('X-RateLimit-Reset'))

        rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 1))

        # If rate limit has been reached...
        if rate_limit_remaining == 0 or response.status_code == 429:
            if response.status_code == 429:
                print(f'Failed to retrieve toots due to {response.status_code} "Too Many Requests" error')

            reset_time_str = response.headers.get('X-RateLimit-Reset')
            reset_time = datetime.strptime(reset_time_str, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
            current_time = datetime.now(timezone.utc)
            wait_time = (reset_time - current_time).total_seconds()

            # Wait until the rate limit resets, plus the buffer time.
            print(f"You've reached the rate limit. Please wait for {wait_time + buffer_time} seconds.")
            time.sleep(wait_time + buffer_time)

            continue

        data = response.json()
        if data:
            last_toot_time = datetime.strptime(data[-1]['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(
                tzinfo=timezone.utc)
            print(f"First toot time: {data[0]['created_at']}, Last toot time: {last_toot_time}")

            new_toots = []
            for toot in data:

                try:
                    created_at = datetime.strptime(toot['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(
                        tzinfo=timezone.utc)
                    if created_at.year <= 1970:
                        raise ValueError
                except ValueError:
                    # If the year is 1970 or in case of another ValueError, use a placeholder date '2000-01-01'.
                    created_at = datetime(2000, 1, 1, tzinfo=timezone.utc)

                try:
                    account_created_at = datetime.strptime(toot['account']['created_at'],
                                                           '%Y-%m-%dT%H:%M:%S.%fZ').replace(
                        tzinfo=timezone.utc)
                    if account_created_at.year <= 1970:
                        raise ValueError
                except ValueError:
                    # If the year is 1970 or in case of another ValueError, use a placeholder date '2000-01-01'.
                    account_created_at = datetime(2000, 1, 1, tzinfo=timezone.utc)

                # Checks if the 'created_at' timestamp of a toot falls within the requested date range.
                if lower_bound_date <= created_at <= upper_bound_date:

                    if lower_bound_date <= account_created_at <= upper_bound_date:
                        # If the account was created within the time period, save the whole toot.
                        new_toots.append(toot)
                    else:
                        # If the account was not created within the time period, save only selected attributes.
                        toot_copy = {key: toot[key] for key in ['id', 'created_at']}
                        toot_copy['account'] = {key: toot['account'].get(key) for key in
                                                ['id', 'username', 'acct', 'display_name', 'created_at', 'note',
                                                 'followers_count', 'following_count', 'statuses_count',
                                                 'last_status_at']}
                        new_toots.append(toot_copy)

            if new_toots:
                print('The toots are within the datetime range. Saving the batch on disk')
                save_results(instance_name, new_toots, index)
                index += 1

            # Update max_id for the next request.
            max_id = data[-1]['id']
            save_checkpoint(instance_name, str(last_toot_time), max_id, index)

        else:
            break

def do(instance_name: str) -> None:
    """
    Main function to initiate the process of toot retrieval.

    :param instance_name: str, instance name whose toots you want to retrieve
    :return: None
    """
    fetch_and_save_toots(instance_name, LOWER_BOUND_DATE, UPPER_BOUND_DATE)

    # Finally we can merge all the batches in a single JSON file.
    # data_mng.merge_json_files(config.DATA_PUBLIC_BATCHES_FOLDER, config.DATA_FOLDER, f'{instance_name}_toots.json')
