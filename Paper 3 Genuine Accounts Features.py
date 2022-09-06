"""
This python file is used to
    - Extract the tweet features of genuine accounts
"""

import csv
import os
import statistics
import numpy
import numpy as np
from Levenshtein import distance as lev
from datetime import *
import re
import pandas as pd


# THE BELOW PIECE OF CODE IS USED TO EXTRACT THE NECESSARY ATTRIBUTES OF THE USER TYPES' TWEETS TO DO FURTHER OPERATIONS ON THEM
# SINCE THE TWEETS FILES ARE HUGE, WE'RE DROP MOST OF THE UNNECESSARY ATTRIBUTES TO CLEAR SOME OVERHEAD
# WE FIRST GET THE GENUINE ACCOUNTS' TWEETS FROM THE CSV FILE, EXTRACT THE NECESSARY TWEET ATTRIBUTES TO ANOTHER DATAFRAME,
# CHANGE TWEET TEXT VALUES TO STRINGS AND FINALLY SAVE THE FILES THAT HAVE THE ATTRIBUTES WE NEED
"""genuineAccountsTweets = pd.read_csv(str(os.getcwd()) + '/cresci-2017/Genuine_Accounts/tweets.csv', low_memory=False)
genuineAccountsTweets = genuineAccountsTweets[
    ["id", "user_id", "text", "retweet_count", 'favorite_count', "created_at", "num_hashtags", "num_urls", "num_mentions"]]
genuineAccountsTweets = genuineAccountsTweets.astype({"text": 'string'})
genuineAccountsTweets.to_csv(str(os.getcwd()) + '/My Model/Genuine Accounts Tweets.csv', index=False)  # All tweets of social spam bots"""

genuineAccountsTweets = pd.read_csv(str(os.getcwd()) + '/My Model/Genuine Accounts Tweets.csv', low_memory=False)  # Get the tweets of the genuine accounts, reduced attribute version
genuineAccountsTweets = genuineAccountsTweets.astype({"text": 'string'})  # Change the tweet texts to string
genuineAccountsTweets['text'] = genuineAccountsTweets['text'].fillna(value='')  # Fill all empty values in the tweet texts with empty string
genuineAccounts = pd.read_csv(str(os.getcwd()) + '/cresci-2017/Genuine_Accounts/users.csv')  # Get the users that are genuine accounts
genuineAccounts['label'] = 0  # Label the humans
goalGenuineAccounts = genuineAccounts[['label', 'id']]  # Create dataframe with id and labels


def extractTweetFeatures(traditionalFinalFeatures):
    totalTweetsCount = 0  # Initialize the total tweet count of the user
    totalTweetSize = 0  # Initialize the total tweet size of the user
    totalHashtagCount = 0  # Initialize the total hashtag count of the user
    totalRetweetCount = 0  # Initialize the total retweeted tweets count of the user
    totalUrlCount = 0  # Initialize the total URL count of the user
    totalMentionCount = 0  # Initialize the total mention count of the user
    totalRetweetToTweetsCount = 0  # Initialize the total retweets the user has for their tweets
    totalLikesToTweetsCount = 0  # Initialize the total favorites the user has for their tweets
    totalTimeBetweenTweets = 0  # Initialize the total time the user has between two tweet
    TTISD = 0  # Initialize TTISD feature
    totalHoursOfTweets = 0  # Initialize the total count for the hours the user tweets
    TTSD = 0  # Initialize TTSD feature
    # Initialize array for the creation times of the tweets
    tweetTimesArray = []  # Time Format: '2010-01-23 17:21:12', '2010-01-22 14:34:26',
    tweetTimeIntervalsArray = []  # Initialize array for tweet time interval
    tweetHoursInDay = []  # Initialize array for the hours the tweets are created
    # Sample all tweets of the user using the ER diagram connection given in the final report
    sample = genuineAccountsTweets.loc[genuineAccountsTweets['user_id'] == traditionalFinalFeatures['id']]
    for _, data in sample.iterrows():   # For all tweets
        totalTweetsCount += 1  # Increase tweet count
        tweetText = data['text']  # Get tweet text
        totalTweetSize += len(tweetText)  # Add tweet size to total tweet size
        totalHashtagCount += data['num_hashtags']  # Add hashtag count to total hashtag count
        totalUrlCount += data['num_urls']  # Add URL count to total URL count
        totalMentionCount += data['num_mentions']  # Add mention count to total mention count
        totalRetweetToTweetsCount += data['retweet_count']  # Add retweet count to the total retweets the user has
        totalLikesToTweetsCount += data['favorite_count']  # Add favorites to the total favorites the user has
        tweetTimesArray.append(data['created_at'])  # Append creation time of the tweet to the times array
        if tweetText[0:4] == 'RT @':  # If the tweet is a retweet
            totalRetweetCount += 1  # Increase the retweet count of the user

    if len(tweetTimesArray) >= 4:   # If the user has more than 4 tweets
        firstTweetTime = datetime.strptime(tweetTimesArray[0], "%a %b %d %X %z %Y")  # Get the time of the tail tweet
        totalHoursOfTweets += firstTweetTime.hour  # Add the hour of that tweet to the total hours
        tweetHoursInDay.append(firstTweetTime.hour)  # Append the hour of that tweet to the hours array
        for i in range(1, len(tweetTimesArray)):  # For all remaining tweets
            traverseTweetTime = datetime.strptime(tweetTimesArray[i], "%a %b %d %X %z %Y")  # Get time of the traversing tweet
            totalHoursOfTweets += traverseTweetTime.hour  # Add the hour of that tweet to the total hours
            tweetHoursInDay.append(traverseTweetTime.hour)  # Append the hour of that tweet to the hours array
            timeDifferenceBetweenTweets = firstTweetTime - traverseTweetTime  # Calculate time difference between the latest two tweets
            tweetTimeIntervalsArray.append(timeDifferenceBetweenTweets.total_seconds() / 86400)  # Append the time different between the latest two tweets to the time difference array
            totalTimeBetweenTweets += timeDifferenceBetweenTweets.total_seconds() / 86400  # Add the time diffference between the latest two tweets to total time between tweets
            firstTweetTime = traverseTweetTime  # Make the traverse tweet the tail
        TTISD = statistics.stdev(tweetTimeIntervalsArray)  # Calculate TTISD
        TTSD = statistics.stdev(tweetHoursInDay)  # Calculate TTSD

    traditionalFinalFeatures['total_tweets'] = totalTweetsCount  # Assign statuses count feature
    traditionalFinalFeatures['TTIM'] = 0 if totalTweetsCount == 0 else totalTimeBetweenTweets / totalTweetsCount  # Calculate and assign TTIM feature
    traditionalFinalFeatures['TTISD'] = TTISD  # Assign TTISD feature
    traditionalFinalFeatures['TTM'] = 0 if totalTweetsCount == 0 else totalHoursOfTweets / totalTweetsCount  # Calculate and assign TTM feature
    traditionalFinalFeatures['TTSD'] = TTSD  # Assign TTSD feature
    traditionalFinalFeatures['avg_tweet_size'] = 0 if totalTweetsCount == 0 else totalTweetSize / totalTweetsCount  # Calculate and assign average tweet size feature
    traditionalFinalFeatures['avg_hashtag_count'] = 0 if totalTweetsCount == 0 else totalHashtagCount / totalTweetsCount  # Calculate and assign average hashtag count feature
    traditionalFinalFeatures['retweeted_to_total_tweets'] = 0 if totalTweetsCount == 0 else totalRetweetCount / totalTweetsCount  # Calculate and assign retweet ratio feature
    traditionalFinalFeatures['url_ratio'] = 0 if totalTweetsCount == 0 else totalUrlCount / totalTweetsCount   # Calculate and assign average URL per tweet feature
    traditionalFinalFeatures['mentions_ratio'] = 0 if totalTweetsCount == 0 else totalMentionCount / totalTweetsCount  # Calculate and assign average mentions per tweet feature
    traditionalFinalFeatures['retweet_per_tweet'] = 0 if totalTweetsCount == 0 else totalRetweetToTweetsCount / totalTweetsCount  # Calculate and assign retweets per tweet feature
    traditionalFinalFeatures['favourites_per_tweet'] = 0 if totalTweetsCount == 0 else totalLikesToTweetsCount / totalTweetsCount  # Calculate and assign favorites per tweet feature

    return traditionalFinalFeatures  # Return feature set


goalGenuineAccounts = goalGenuineAccounts.apply(extractTweetFeatures, axis=1)  # Apply feature extraction to all labeled users
# goalGenuineAccounts = goalGenuineAccounts.drop(['id'], axis=1)
goalGenuineAccounts.to_csv(str(os.getcwd()) + '/My Model/Genuine Accounts Tweet Features ID.csv', index=False)  # Save tweet features of the user type

