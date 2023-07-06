import os

# os.getcwd() gets as root directory the same folder in which the main run script (run.py) is located

DATA_FOLDER = os.getcwd() + '/mastodon/data/'
DATA_PUBLIC_BATCHES_FOLDER = DATA_FOLDER + 'public_batches/'
DATA_INSTANCES_FOLDER = DATA_FOLDER + 'instances/'
DATA_USER_BATCHES_FOLDER = DATA_FOLDER + 'user_batches/'
DATA_SQL_OUTPUT_FOLDER = DATA_FOLDER + 'SQL_output/'
MODELS_FOLDER = os.getcwd() + '/mastodon/models/'
LOGS_FOLDER = os.getcwd() + '/mastodon/logs/'
DTM_MODELS_FOLDER = MODELS_FOLDER + 'dtm/'