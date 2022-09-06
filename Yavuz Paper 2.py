import pandas as pd
from datetime import *
import emoji
import re

dataset = pd.read_csv('accounts_with_tweets.csv')

goal_dataset = dataset[["id", "label"]]

tweet_dataset = pd.read_csv('tweet_dataset.csv')





def extract_features(df):

    retweet_count = 0
    tweet_count = 0
    reply_count = 0
    favorite_count = 0
    number_of_hashtags = 0
    number_of_urls = 0
    number_of_mentions = 0
    friends_count = 0 
    followers_count = 0
    posting_time_array = []
    posting_time_difference = 0

    samples = tweet_dataset.loc[tweet_dataset['user_id'] == df['id']]
    for _, data in samples.iterrows():
        posting_time_array.append(data['created_at'])
        if type(data['text']) == str and (data['text'].startswith("RT @")): 
            retweet_count += 1

        else:
            tweet_count += 1
            if data["in_reply_to_status_id"] != 0:
                reply_count += 1
            number_of_hashtags += data["num_hashtags"]
            number_of_urls += data["num_urls"]
            number_of_mentions += data["num_mentions"]

    user = dataset.loc[dataset['id'] == df['id']]
    for _, data in user.iterrows():
        favorite_count = data["favourites_count"]
        friends_count = data["friends_count"]
        followers_count = data["followers_count"]


    for x in range(retweet_count + tweet_count -1):
        if posting_time_array[x+1][0].isdigit():
            time1 = datetime.strptime(posting_time_array[x+1],"%Y-%m-%d %X")
            time2 = datetime.strptime(posting_time_array[x],"%Y-%m-%d %X")
        else:
            time1 = datetime.strptime(posting_time_array[x+1],"%a %b %d %X %z %Y")
            time2 = datetime.strptime(posting_time_array[x],"%a %b %d %X %z %Y")       
        
        diff = time2 - time1
        posting_time_difference += diff.total_seconds()

    df["retweets"] = 0 if tweet_count < 1 else retweet_count / tweet_count
    df["replies"] = 0 if tweet_count < 1 else reply_count / tweet_count
    df["favoriteC"] = 0 if tweet_count < 1 else favorite_count / tweet_count
    df["hashtag"] = 0 if tweet_count < 1 else number_of_hashtags / tweet_count
    df["url"] = 0 if tweet_count < 1 else number_of_urls / tweet_count
    df["mentions"] = 0 if tweet_count < 1 else number_of_mentions / tweet_count
    df["intertime"] = 0 if tweet_count + retweet_count < 1 else posting_time_difference / (tweet_count + retweet_count)
    df["ffratio"] = 0 if followers_count < 1 else friends_count / followers_count
    df["favorites"] = favorite_count
    return df




goal_dataset = goal_dataset.apply(extract_features, axis=1)

goal_dataset.to_csv("paper2_features.csv")