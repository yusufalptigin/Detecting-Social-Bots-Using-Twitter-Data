# For sending GET requests from the API
import requests
# For saving access tokens and for file management when creating and adding to the dataset
import os
# For dealing with json responses we receive from the API
import json
# For displaying the data after
import pandas as pd
# For saving the response data in CSV format
import csv
# For parsing the dates received from twitter in readable formats
import datetime
import dateutil.parser
import unicodedata
# To add wait time between requests
import time

os.environ['TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAAFzGcgEAAAAAGYAEB3co11SkDVxBeAD9vYJjQr4%3DVfr86gqFSDV41twBffRdIoJxQ4iDR4ox9ZTHl5YdzuEtU7MuJG'
bearer_token = os.getenv('TOKEN')
headers = {"Authorization": "Bearer {}".format(bearer_token)}


def recent_Search_Endpoint(keyword, max_results=100):
    endpointUrl = "https://api.twitter.com/2/tweets/search/recent"  # Change to the endpoint you want to collect data from

    # change params based on the endpoint you are using
    query_params = {'query': keyword,
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return endpointUrl, query_params


def connect_to_endpoint(endpointUrl, Headers, params, next_token=None):
    params['next_token'] = next_token  # params object received from create_url function
    response = requests.request("GET", endpointUrl, headers=Headers, params=params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


"""query = "from:1473155256"
url = recent_Search_Endpoint(query, 100)
json_response = connect_to_endpoint(url[0], headers, url[1])
print(json.dumps(json_response, indent=4, sort_keys=True))
df = pd.DataFrame(json_response['data'])
df.to_csv('Tweets.csv')"""
