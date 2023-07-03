# Importing required Libraries

import requests
import pandas as pd
import numpy as np
import json


base_url = 'https://www.climatewatchdata.org/api/v1/data/historical_emissions?source_ids[]=190&gas_ids[]=442&page=1&per_page=200&sector_ids[]=2214&sort_col=2020&sort_dir=DESC'
# Requesting data from the API

response = requests.get(base_url)

# Converting the response to JSON format

data = response.json()

# Extracting the data from the JSON format

data = data['data']

# file path to save the data

file_path = 'data/raw/data.json'

# Write the data to the JSON file

''' Saving data in json format as using it from here has an advantage of no need to 
request the data again and again from the API. '''

with open(file_path, "w") as json_file:
    json.dump(data, json_file)
