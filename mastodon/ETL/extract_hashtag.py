import requests
import random
from urllib.parse import urlparse
import json
from datetime import datetime, timezone

from mastodon.settings import config
from mastodon.utils import data_mng

# Replace ACCESS_TOKEN with your actual access token (it's not necessary in order to interact with the public timeline
# API).
ACCESS_TOKEN = 'YOUR_ACCESS_TOKEN'

# Define the list of Mastodon instance URLs.
BASE_URLS = ['https://sciences.social', 'https://mastodon.social', 'https://mstdn.social', 'https://mastodon.world',
             'https://mas.to', 'https://mastodon.online', 'https://mstdn.party', 'https://mastodon.xyz']

"""
https://medium.com/@mastodonmigration/sharing-advice-and-assisting-with-the-great-mastodon-migration-53c1a286b805#cfd1
Hashtags:
    #twittermigration #mastodonmigration

Data structure:    
    https://mastodonpy.readthedocs.io/en/1.4.3/
"""

# partial URL for the API endpoint that retrieves timelines based on a specific tag or hashtag.
API_ENDPOINT = '/api/v1/timelines/tag/'

# Lower bound retrieval date: No data older than this date will be extracted.
CUT_OFF_DATE = '2022-04-01'

def make_request(base_url: str, endpoint: str, max_id: str, since_id: str) -> requests.models.Response:
    """
    Makes a GET request to the Mastodon API.

    :param base_url: A string representing the base URL of the Mastodon instance.
    :param endpoint: A string representing the API endpoint.
    :param max_id: A string representing the maximum ID of the items to include in the response.
    :param since_id: A string representing the minimum ID of the items to include in the response.
    :return: A Response object representing the server's response to the request.
    """
    return requests.get(
        f"{base_url}{endpoint}",
        params={
            'max_id': max_id,
            'since_id': since_id,
            'local': 'false'
        },
        headers={
            'Authorization': f'Bearer {ACCESS_TOKEN}',
            'Content-Type': 'application/json'
        }
    )


from datetime import datetime, timezone
from typing import List, Dict

def check_last_toot_date(data: list, cut_off_date: str) -> bool:
    """
    Check if the date of the last toot in the data list is older than the specified cut-off date.

    :param data: A list of toots, where each toot is represented as a dictionary.
    :param cut_off_date: A string representing the cut-off date in the format 'YYYY-MM-DD'.
    :return: False if the last toot is older than the cut-off date, None otherwise.
    """
    # Convert the cut-off date to a datetime object for comparison.
    cut_off_date = datetime.strptime(cut_off_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    last_toot_time = datetime.strptime(data[-1]['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(
        tzinfo=timezone.utc)
    print(f"First toot time: {data[0]['created_at']}, Last toot time: {last_toot_time}")

    # Check if the last toot is older than the cut-off date
    if last_toot_time < cut_off_date:
        print('The toots are now older than the cut-off date. Stopping the retrieval process.')
        return False

    return True

def process_and_save_data(data: list, base_url: str, hashtag: str) -> bool:
    """
    Processes the data returned from the API and saves it to a JSON file.

    :param data: A list of dictionaries representing the data returned from the API.
    :param base_url: A string representing the base URL of the Mastodon instance.
    :param hashtag: A string representing the current hashtag we're gathering the toots from.
    :return: None
    """
    # Iterate over each JSON object in the data list.
    for item in data:
        # Add the 'queried_instance' key-value pair.
        item['queried_instance'] = urlparse(base_url).hostname

    # Save the data to a JSON file.
    filename = f'{config.DATA_FOLDER}{hashtag}_toots.json'
    data_mng.save_list_to_json(data, filename, 'a')

    return True

def do(hashtag: str) -> None:
    """
    Main function to fetch and process toots from the Mastodon API.

    :return: None
    """
    # Set the maximum and minimum IDs of the items to include in the response.
    max_id = None
    since_id = None

    # Set the API endpoint for retrieving the public timeline.
    endpoint = API_ENDPOINT + hashtag

    while True:

        # Choose a random Mastodon instance URL from the list.
        # Randomizing the istance from which retrive the data should prevent reaching the rate limit.
        base_url = random.choice(BASE_URLS)

        # Make a GET request to the API endpoint
        response = make_request(base_url, endpoint, max_id, since_id)

        # Check the response status code to see if the request was successful.
        if response.status_code == 200:

            # The request was successful, so parse the response as JSON.
            data = response.json()

            if data:
                # Update the maximum and minimum IDs for the next request.
                # The max_id parameter should be set to the ID of the oldest item in the list
                # (otherwise, the since_id parameter should be set to the ID of the newest item in the list).
                max_id = data[-1]['id']

                print(f"Fetched toots up to {data[-1]['created_at']}, and id {max_id}, from {base_url}...")

                # Process and save to disk the batch of toots.
                process_and_save_data(data, base_url, hashtag)

                print('Rate limit:', response.headers.get('X-RateLimit-Limit'))
                print('Remaining:', response.headers.get('X-RateLimit-Remaining'))
                print('Reset time:', response.headers.get('X-RateLimit-Reset'))

                if not check_last_toot_date(data, CUT_OFF_DATE):
                    break

            else:
                print('No response data available. Exiting the loop.')
                break

            # Check if there are more pages of results.
            if 'next' in response.links:
                continue
            else:
                print('No more pages of results. Exiting the loop.')
                break
        else:
            print(f'Failed to retrieve toots from the public timeline: {response.status_code}')
            break