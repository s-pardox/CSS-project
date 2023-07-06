import os
import requests
from datetime import datetime, timedelta
import base64

import requests
import time
from requests.models import Response

from mastodon.settings import config

def handle_rate_limit(response: Response) -> bool:
    """
    Handle the rate limit from the GitHub API by sleeping the script until the rate limit resets.

    :param response: The Response object from a request to the GitHub API.
    :return: Boolean indicating whether the rate limit was encountered or not.
    """
    if response.status_code == 429:  # HTTP Status code for Too Many Requests
        rate_limit_reset = int(response.headers['X-RateLimit-Reset'])
        current_time = int(time.time())
        sleep_time = rate_limit_reset - current_time + 5  # Add a small buffer to ensure the rate limit has reset
        print(f"Rate limit exceeded. Sleeping for {sleep_time} seconds.")
        time.sleep(sleep_time)
        return True
    return False

def get_commit_files_in_date_range(owner: str, repo: str, github_token: str, start_date: str, end_date: str) -> list:
    """
    Fetches commit files from the GitHub repository using GitHub's API v3.

    :param owner: A string representing the owner of the repository.
    :param repo: A string representing the repository name.
    :param github_token: A string representing the personal access token for GitHub API access.
    :param start_date: A string representing the start date in 'YYYY-MM-DD' format.
    :param end_date: A string representing the end date in 'YYYY-MM-DD' format.
    :return: A list of commit file information where each entry is a dict with 'date' and 'url' keys.
    """
    headers = {"Authorization": f"token {github_token}"}
    params = {"path": "instances.json", "page": 1}
    commit_files = []
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    # We need to include both dates.
    end_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)

    while True:
        while True:
            print(f"Fetching commits from page {params['page']}...")
            response = requests.get(f"https://api.github.com/repos/{owner}/{repo}/commits", headers=headers, params=params)

            # If a rate limit was not encountered, break the loop
            if not handle_rate_limit(response):
                break

        data = response.json()

        # Check if 'data' is a list and is not empty before accessing its elements
        if isinstance(data, list) and data:
            first_commit_date = datetime.strptime(data[0]['commit']['committer']['date'], '%Y-%m-%dT%H:%M:%SZ')
            last_commit_date = datetime.strptime(data[-1]['commit']['committer']['date'], '%Y-%m-%dT%H:%M:%SZ')
            print(f"Time range on this page: {first_commit_date} to {last_commit_date}")

            for commit in data:
                commit_date = datetime.strptime(commit['commit']['committer']['date'], '%Y-%m-%dT%H:%M:%SZ')

                # Filter commits within the specified date range
                if start_date <= commit_date < end_date:
                    try:
                        file_url = get_file_url_from_tree(commit['commit']['tree']['url'], 'instances.json',
                                                          github_token)
                        if file_url:
                            commit_files.append({'date': commit_date, 'url': file_url})
                            print(f"Found commit from {commit_date} within specified date range.")
                    except KeyError as error:
                        print(f"Key 'tree' not found in commit: {commit}")
                        log_error(error, commit, start_date, end_date)
        else:
            if not data:
                print("No more commits found.")
                break
            else:
                print(f"Unexpected data structure received: {data}")
                break
        params["page"] += 1

    return commit_files

def get_first_commit_files_each_day(commit_files: list) -> list:
    """
    Filters the commit files to include only the first commit file of each day.
    :param commit_files: A list of commit file information where each entry is a dict with 'date' and 'url' keys.
    :return: A filtered list of commit file information including only the first commit file of each day.
    """
    filtered_commit_files = []
    dates_seen = set()

    for file in sorted(commit_files, key=lambda x: x['date']):
        date = file['date'].date()

        # Only add the commit file info if we haven't seen a commit from this date yet.
        if date not in dates_seen:
            filtered_commit_files.append(file)
            dates_seen.add(date)

    return filtered_commit_files

def get_last_commit_files_each_day(commit_files: list) -> list:
    """
    Filters the commit files to include only the last commit file of each day.
    :param commit_files: A list of commit file information where each entry is a dict with 'date' and 'url' keys.
    :return: A filtered list of commit file information including only the last commit file of each day.
    """
    filtered_commit_files = []
    dates_seen = set()

    for file in sorted(commit_files, key=lambda x: x['date'], reverse=True):
        date = file['date'].date()

        # Only add the commit file info if we haven't seen a commit from this date yet.
        if date not in dates_seen:
            filtered_commit_files.append(file)
            dates_seen.add(date)

    # Revert the order of the list to return entries in ascending date order
    return list(reversed(filtered_commit_files))

def download_file(download_url: str, output_path: str, github_token: str):
    """
    Downloads a file from the given URL.

    :param download_url: A string representing the URL of the file.
    :param output_path: A string representing the path where the downloaded file should be saved.
    :param github_token: A string representing the personal access token for GitHub API access.
    """
    headers = {"Authorization": f"token {github_token}"}

    while True:
        print(f"Downloading file from URL: {download_url}")
        response = requests.get(download_url, headers=headers)

        # If a rate limit was not encountered, break the loop
        if not handle_rate_limit(response):
            break

    data = response.json()

    if 'content' in data:
        with open(output_path, 'w') as file:
            file_content = base64.b64decode(data['content']).decode('utf-8')
            file.write(file_content)
        print(f"Successfully downloaded file to path: {output_path}")
    else:
        print(f"'content' key not found in data: {data}")

def get_file_url_from_tree(tree_url: str, filename: str, github_token: str) -> str:
    """
    Fetches the URL of the specified file from the GitHub tree structure.

    :param tree_url: A string representing the URL of the GitHub tree structure.
    :param filename: A string representing the name of the file to find.
    :param github_token: A string representing the personal access token for GitHub API access.
    :return: A string representing the URL of the specified file.
    """
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(tree_url, headers=headers)
    data = response.json()

    for file_info in data['tree']:
        if file_info['path'] == filename:
            return file_info['url']

    return None

def log_error(error, commit, start_date, end_date):
    """
    Logs error details to a text file.

    :param error: The error object.
    :param commit: The commit dict at the time of error.
    :param start_date: The start date of the range.
    :param end_date: The end date of the range.
    """
    filename = f'{config.DATA_FOLDER}instances_retrieval_error_log.txt'

    with open(filename, 'a') as f:
        f.write(f"Error: {str(error)}\n")
        f.write(f"Commit SHA: {commit['sha']}\n")
        f.write(f"Commit Author: {commit['commit']['author']['name']}\n")
        f.write(f"Commit Date: {commit['commit']['committer']['date']}\n")
        f.write(f"Date Range: {start_date} to {end_date}\n")
        f.write("\n")

def do(owner: str, repo: str, github_token: str, start_date: str, end_date: str, output_folder: str):
    """
    Fetches and downloads GitHub commits in a given date range for a specific repository.

    :param owner: A string representing the owner of the GitHub repo.
    :param repo: A string representing the name of the GitHub repo.
    :param github_token: A string representing the GitHub access token.
    :param start_date: A string representing the start date in 'YYYY-MM-DD' format.
    :param end_date: A string representing the end date in 'YYYY-MM-DD' format.
    :param output_folder: A string representing the output directory where the downloaded files should be saved.
    """
    print("Fetching commit files from GitHub API...")
    commit_files = get_commit_files_in_date_range(owner, repo, github_token, start_date, end_date)

    print(f"Fetched {len(commit_files)} commit files. Filtering for first commit of each day...")
    commit_files = get_last_commit_files_each_day(commit_files)

    print(f"Filtered to {len(commit_files)} commit files. Starting to download commit files...")
    for commit_file in commit_files:
        date_str = commit_file['date'].strftime('%Y-%m-%d_%H-%M-%SZ')
        download_file(commit_file['url'], os.path.join(output_folder, f"{date_str}.json"), github_token)

    print("Finished downloading commit files.")