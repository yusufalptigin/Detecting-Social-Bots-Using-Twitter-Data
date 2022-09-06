"""
This python file is used to:
    - Concatenate the tweet features of all types of users
    - Merge the account and tweet features of all users
"""

import os
import pandas as pd

# THIS PORTION OF THE CODE IS USED TO CONCATENATE THE TWEET FEATURES OF ALL TYPES OF USERS
# IT IS COMMENTED TO DECREASE OVERHEAD
# SHOULD BE COMMENTED OUT WHEN USED

"""fakeFollowerTweetFeatures = pd.read_csv(str(os.getcwd()) + '/My Model/Fake Followers Tweet Features ID.csv')  # Get tweet features of fake followers
genuineAccountsTweetFeatures = pd.read_csv(str(os.getcwd()) + '/My Model/Genuine Accounts Tweet Features ID.csv')  # Get tweet features genuine accounts
socialBotsTweetFeatures = pd.read_csv(str(os.getcwd()) + '/My Model/Social Bots Tweet Features ID.csv')   # Get tweet features social bots
traditionalBotsTweetFeatures = pd.read_csv(str(os.getcwd()) + '/My Model/Traditional Bots Tweet Features ID.csv')   # Get tweet features traditional bots

frames = [genuineAccountsTweetFeatures, fakeFollowerTweetFeatures, socialBotsTweetFeatures, traditionalBotsTweetFeatures]  # Put tweet features into a frame array

allTweetFeatures = pd.concat(frames)  # Concatenate tweet features of all types of users
allTweetFeatures.to_csv(str(os.getcwd()) + '/My Model/All Tweets Features ID.csv', index=False)  # Save tweet features
"""

allTweetFeatures = pd.read_csv(str(os.getcwd()) + '/My Model/All Tweets Features ID.csv')  # Get tweet features
allUserFeatures = pd.read_csv(str(os.getcwd()) + '/My Model/All Users Features ID.csv')  # Get user features

mergedDataset = pd.merge(allTweetFeatures, allUserFeatures, on='id', how='outer')  # Merge all features by id
mergedDataset.to_csv(str(os.getcwd()) + '/My Model/Paper 3 All Features.csv', index=False)  # Save all features
mergedDataset = mergedDataset.drop(['id'], axis=1)  # Drop id label since it's string
mergedDatasetFiltered = mergedDataset[mergedDataset['total_tweets'] >= 4]  # Filter all users with less than 4 tweets
mergedDatasetFiltered.to_csv(str(os.getcwd()) + '/My Model/Paper 3 All Features Filtered.csv', index=False)  # Save all users with their features after filtering
