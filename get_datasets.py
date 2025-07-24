'''
Allows to get datasets from UCI Machine Learning repositorys. Saves them in /ds as ids in a json file.
'''

import requests
import json
import os

API_LIST_URL = 'https://archive.ics.uci.edu/api/datasets/list'
def get_datasets():
    response = requests.get(API_LIST_URL)
    if response.status_code == 200:
        datasets = response.json()
        dataset_ids = [dataset['id'] for dataset in datasets['data']]
        if not os.path.exists('ds'):
            os.makedirs('ds')
        with open('ds/dataset_ids.json', 'w') as f:
            json.dump(dataset_ids, f, indent=4)
        print(f"Saved {len(dataset_ids)} dataset IDs to ds/dataset_ids.json")
    else:
        print(f"Failed to retrieve datasets. Status code: {response.status_code}")

if __name__ == "__main__":
    get_datasets()
