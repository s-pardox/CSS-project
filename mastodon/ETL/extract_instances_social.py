import requests
import json
from typing import List, Dict

def get_top_instances(token: str, count: int = 10) -> List[Dict]:
    """
    Retrieve the top populated Mastodon instances.

    :param token: The API token.
    :type token: str
    :param count: The number of instances to retrieve. Defaults to 10.
    :type count: int, optional
    :return: A list of dictionaries where each dictionary contains information about an instance.
    :rtype: List[Dict]

    Example usage:
        instances = get_top_instances(token='YOUR_TOKEN', count=10)
        for instance in instances:
            print(f"Instance: {instance['name']}, Users: {instance['users']}")
    """

    # Set the headers.
    headers = {'Authorization': f'Bearer {token}'}

    # Send a GET request to the 'instances/list' endpoint.
    response = requests.get('https://instances.social/api/1.0/instances/list', headers=headers)

    # Print the status code and the response text.
    print(f'Status code: {response.status_code}')
    print(f'Response text: {response.text}')

    # Load the JSON response into a Python object.
    instances = json.loads(response.text)['instances']

    # Sort the instances by user count in descending order and keep the top instances.
    top_instances = sorted(instances, key=lambda i: int(i['users']), reverse=True)[:count]

    return top_instances