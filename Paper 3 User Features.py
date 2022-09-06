"""
This python file is used to
    - Extract the user features of all users
"""

import csv
import os
from Levenshtein import distance as lev
from datetime import *
import re
import pandas as pd


# Create string length function
def string_length(x):  # returns string length
    return len(x)


# Create LD feature function
def lev_distance(userName, screenName):
    return lev(userName, screenName)


# Create AA feature function
def yearlyAge(createDateString):
    if createDateString[len(createDateString) - 1] == 'L':
        return 13
    else:
        return 2022 - int(createDateString[(len(createDateString) - 5): (len(createDateString))])


# Create FFR feature function
def followersToFriendsRatio(followers_count, friends_count):
    if followers_count == 0 or friends_count == 0:
        return 0
    else:
        return followers_count / friends_count


# Create RS feature function
def reputationScore(followers_count, friends_count):
    if followers_count == 0 or friends_count == 0:
        return 0
    else:
        return followers_count / (followers_count + friends_count)


# THE BELOW PIECE OF CODE IS USED TO GET ALL USER ACCOUNTS INTO DIFFERENT DATAFRAMES TO THEN CONCATENATE
"""genuineUsers = pd.read_csv(str(os.getcwd()) + '/cresci-2017/Genuine_Accounts/users.csv')  # Get genuine users
genuineUsers = genuineUsers.drop(columns=['test_set_1', 'test_set_2'])  # Drop unnecessary columns
genuineUsers['label'] = 0  # Label user

fakeFollowers = pd.read_csv(str(os.getcwd()) + '/cresci-2017/Fake_Followers/users.csv')  # Get fake follower bots
fakeFollowers['label'] = 1  # Label user

socialBots = pd.read_csv(str(os.getcwd()) + '/cresci-2017/Social_Spambots/users.csv')  # Get social bots
socialBots['label'] = 1  # Label user

traditionalBots = pd.read_csv(str(os.getcwd()) + '/cresci-2017/Traditional_Spambots/users.csv')  # Get traditional bots
traditionalBots['label'] = 1  # Label user

frames = [genuineUsers, fakeFollowers, socialBots, traditionalBots]  # Get all datasets into an array
allUsers = pd.concat(frames)  # Concatenate all users
allUsers.to_csv(str(os.getcwd()) + '/My Model/All Users.csv', index=False)  # Save all users"""

allUsersCSV = pd.read_csv(str(os.getcwd()) + '/My Model/All Users.csv')  # Get all users
# Create a dataframe and get all the necessary user attributes to either directly use as a feature or derive features from them
allUsersFeatures = allUsersCSV[['description', 'created_at', 'name', 'screen_name', 'geo_enabled', 'statuses_count',
                                'followers_count', 'friends_count', 'favourites_count', 'label']]
allUsersFeatures = allUsersFeatures.astype({'name': 'string'})  # Change user name types to string
allUsersFeatures = allUsersFeatures.astype({'screen_name': 'string'})  # Change screen name types to string
allUsersFeatures['geo_enabled'] = allUsersFeatures['geo_enabled'].fillna(value=0)  # Fill null values with 0
allUsersFeatures['name'] = allUsersFeatures['name'].fillna(value='')  # Fill null values with empty string
allUsersFeatures['description'] = allUsersFeatures['description'].fillna(value='')  # Fill null values with empty string
allUsersFeatures['user_name_length'] = allUsersFeatures['name'].apply(string_length)  # Calculate and assign username length feature
allUsersFeatures['screen_name_length'] = allUsersFeatures['screen_name'].apply(string_length)  # Calculate and assign screen name length feature
allUsersFeatures['description_length'] = allUsersFeatures['description'].apply(string_length)  # Calculate and assign description length feature
# Calculate and assign FFR feature
allUsersFeatures['follower_to_friends_ratio'] = allUsersFeatures.apply(lambda x: followersToFriendsRatio(x['followers_count'], x['friends_count']), axis=1)
# Calculate and assign RS feature
allUsersFeatures['reputation_score'] = allUsersFeatures.apply(lambda x: reputationScore(x['followers_count'], x['friends_count']), axis=1)
# Calculate and assign LD feature
allUsersFeatures['lev_distance'] = allUsersFeatures.apply(lambda x: lev_distance(x['name'], x['screen_name']), axis=1)
# Calculate and assign AA feature
allUsersFeatures['age'] = allUsersFeatures['created_at'].apply(yearlyAge)
# Calculate and assign TCAR feature
allUsersFeatures['tweet_to_age_ratio'] = allUsersFeatures['statuses_count'] / allUsersFeatures['age']
# Drop unnecessary columns
allUsersFeatures = allUsersFeatures.drop(['created_at', 'name', 'screen_name', 'description'], axis=1)  # Drop unnecessary columns
# Save all user features
allUsersFeatures.to_csv(str(os.getcwd()) + '/My Model/All Users Features.csv', index=False)  # Save features
