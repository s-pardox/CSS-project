import requests
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

from mastodon.settings import config
from mastodon.settings import mastodon_credentials
from mastodon.utils import data_mng

# Define the base URL of the Mastodon instance.
BASE_URL = 'https://mastodon.social'

# Set the API endpoint for searching for a user.
SEARCH_ENDPOINT = '/api/v1/accounts/search'

# Set the API endpoint for retrieving a user's toots.
USER_ENDPOINT = '/api/v1/accounts/{}/statuses'

# The old official user statistics publisher: @mastodonusercount@bitcoinhackers.org
# (until 2023-02-23 16:00:13.548000+00:00)
OLD_USER_ID = 415114
# Set the username of the user whose toots you want to retrieve (starting from 2023-02-23 17:00:11+00:00)
NEW_USERNAME = 'mastodonusercount'

# Set here your access token or use the one defined in the settings/credentials.py.
ACCESS_TOKEN = mastodon_credentials.ACCESS_TOKEN

# Lower bound retrieval date: No data older than this date will be extracted.
CUT_OFF_DATE = '2022-01-01'

def get_user_id(username: str) -> Optional[str]:
    """
    Search for a user by their username and return their user ID.

    This function queries the Mastodon API's search endpoint with a given username, and returns the user ID of the first
    match, if any is found. If no match is found, it returns None.

    :param username: The username of the user to search for, as a string.
    :return: The user ID of the matched user as a string, or None if no match is found.
    """
    response = requests.get(
        f"{BASE_URL}{SEARCH_ENDPOINT}",
        params={'q': username, 'limit': 1},
        headers={'Authorization': f'Bearer {ACCESS_TOKEN}'}
    )
    data = response.json()
    if data:
        return data[0]['id']
    else:
        return None

def estimate_toot_rate(user_id: str, num_samples: int = 10) -> int:
    """
    Estimate the number of toots per hour for a given user.

    :param user_id: str, user id to estimate the toot rate for
    :param num_samples: int, number of samples to use for the estimate
    :return: int, estimated number of toots per hour
    """
    response = requests.get(
        f"{BASE_URL}{USER_ENDPOINT.format(user_id)}",
        params={'limit': num_samples},
        headers={'Authorization': f'Bearer {ACCESS_TOKEN}'}
    )
    data = response.json()

    if len(data) < 2:
        raise Exception('Not enough toots for estimate.')

    first_toot_time = datetime.strptime(data[0]['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')
    last_toot_time = datetime.strptime(data[-1]['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')

    time_diff = first_toot_time - last_toot_time
    hours_diff = time_diff.total_seconds() / 3600

    toots_per_hour = len(data) / hours_diff

    return round(toots_per_hour)

def save_checkpoint(user_id: str, created_at: str, max_id: int) -> None:
    """
    Store intermediate metadata results into a JSON file.

    :param user_id: str, user id whose toots we're retrieving
    :param created_at: str, the maximum creation datetime reached during the last API interaction
    :param max_id: int, the maximum id of the toots to include in the next request
    :return: None
    """
    filename = f'{config.DATA_FOLDER}{user_id}_checkpoint.json'
    data_mng.save_dict_to_json({'created_at': created_at, 'max_id': max_id}, filename)

def load_checkpoint(user_id: str) -> Optional[Tuple[str, int]]:
    """
    Load any previously saved checkpoints from which retrieval process can restart.

    :param user_id: str, user id whose toots you want to retrieve
    :return: Optional[Tuple[str, int]], a tuple containing the creation time and the maximum id of the toots to include
             in the next request, or None if no checkpoint exists
    """
    filename = f'{config.DATA_FOLDER}{user_id}_checkpoint.json'

    result = data_mng.load_from_json_to_dict(filename)
    if result:
        return result['created_at'], result['max_id']
    else:
        return None

def save_results(user_id: str, data: list, index: int) -> None:
    """
    Store batch results into a JSON file.

    :param user_id: str, the ID of the user whose toots are being saved
    :param data: list, the toots
    :param data: an incremental integer value.
    """
    filename = f'{config.DATA_USER_BATCHES_FOLDER}{user_id}_toots-{index}.json'
    data_mng.save_list_to_json(data, filename)

def fetch_and_save_toots(user_id: str, cut_off_date: str) -> None:
    """
    Retrieve and store a user's 'toots' in a JSON file.
    This function fetches all toots of a given user, starting from the most recent ones,
    and writes them into a JSON file.

    :param user_id: str, user id whose toots you want to retrieve
    :param cut_off_date: str, the date from which we should stop retrieving toots. Format should be 'YYYY-MM-DD'.
    :return: None
    """

    # Convert the cut-off date to a datetime object for comparison.
    cut_off_date = datetime.strptime(cut_off_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    # Are there any previously saved checkpoints from which we can restart the retrieval process?
    checkpoint = load_checkpoint(user_id)
    if checkpoint is not None:
        created_at, max_id = checkpoint
    else:
        max_id = None

    # Index used as suffix in the batch filename.
    index = 0

    while True:
        if max_id is not None:
            print(f'Fetching toots for user {user_id} starting from {created_at}, id {max_id}...')
            response = requests.get(
                f"{BASE_URL}{USER_ENDPOINT.format(user_id)}",
                params={'max_id': max_id, 'limit': 200},
                headers={'Authorization': f'Bearer {ACCESS_TOKEN}'}
            )
        else:
            print(f'Fetching toots for user {user_id}...')
            response = requests.get(
                f"{BASE_URL}{USER_ENDPOINT.format(user_id)}",
                params={'limit': 200},
                headers={'Authorization': f'Bearer {ACCESS_TOKEN}'}
            )

        if response.status_code == 429:
            print(f'Failed to retrieve toots due to {response.status_code} "Too Many Requests" error')
            break

        if response.status_code != 200:
            print(f'Failed to retrieve toots, error {response.status_code}')
            break

        print('Rate limit:', response.headers.get('X-RateLimit-Limit'))
        print('Remaining:', response.headers.get('X-RateLimit-Remaining'))
        print('Reset time:', response.headers.get('X-RateLimit-Reset'))

        rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 1))
        if rate_limit_remaining == 0:
            reset_time_str = response.headers.get('X-RateLimit-Reset')
            reset_time = datetime.strptime(reset_time_str, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
            current_time = datetime.now(timezone.utc)
            wait_time = (reset_time - current_time).total_seconds()
            print(f"You've reached the rate limit. Please wait for {wait_time} seconds.")
            # Wait until the rate limit resets, plus one second to be sure.
            time.sleep(wait_time + 1)
            # Skip the rest of this loop iteration and start the next one.
            continue

        data = response.json()
        if data:
            last_toot_time = datetime.strptime(data[-1]['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(
                tzinfo=timezone.utc)
            print(f"First toot time: {data[0]['created_at']}, Last toot time: {last_toot_time}")

            # Check if the last toot is older than the cut-off date
            if last_toot_time < cut_off_date:
                print('The toots are now older than the cut-off date. Stopping the retrieval process.')
                break

            created_at = data[-1]['created_at']
            max_id = data[-1]['id']

            save_results(user_id, data, index)
            save_checkpoint(user_id, created_at, max_id)
            index += 1

        else:
            break

def do() -> None:
    """
    Main function to initiate the process of toot retrieval.

    :return: None
    """
    new_user_id = get_user_id(NEW_USERNAME)

    # First retrieve the oldest posts.
    fetch_and_save_toots(OLD_USER_ID, CUT_OFF_DATE)

    # Let's retrieve the newer toots.
    if new_user_id is not None:
        fetch_and_save_toots(new_user_id, CUT_OFF_DATE)
    else:
        print(f'User not found: {NEW_USERNAME}')

    # Finally we can merge all the batches in a single JSON file.
    data_mng.merge_json_files(config.DATA_USER_BATCHES_FOLDER, config.DATA_FOLDER, f'{OLD_USER_ID}-{new_user_id}_toots.json')