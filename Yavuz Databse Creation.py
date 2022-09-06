import pandas as pd


users = pd.read_csv('datasets\\cresci-2017.csv\\datasets_full.csv\\genuine_accounts.csv\\genuine_accounts.csv\\users.csv')

users  = users.drop(['test_set_1', 'test_set_2'], axis = 1)

users['label'] = 0

social_spambots_1 = pd.read_csv('datasets\\cresci-2017.csv\\datasets_full.csv\\social_spambots_1.csv\\social_spambots_1.csv\\users.csv')

social_spambots_1  = social_spambots_1.drop(['test_set_1'], axis = 1)

social_spambots_2 = pd.read_csv('datasets\\cresci-2017.csv\\datasets_full.csv\\social_spambots_2.csv\\social_spambots_2.csv\\users.csv')

social_spambots_3 = pd.read_csv('datasets\\cresci-2017.csv\\datasets_full.csv\\social_spambots_3.csv\\social_spambots_3.csv\\users.csv')

social_spambots_3  = social_spambots_3.drop(['test_set_2'], axis = 1)

frames = [social_spambots_1, social_spambots_2, social_spambots_3]

social_spambots = pd.concat(frames)


traditional_spambots_1 = pd.read_csv('datasets\\cresci-2017.csv\\datasets_full.csv\\traditional_spambots_1.csv\\traditional_spambots_1.csv\\users.csv')

traditional_spambots_2 = pd.read_csv('datasets\\cresci-2017.csv\\datasets_full.csv\\traditional_spambots_2.csv\\traditional_spambots_2.csv\\users.csv')

traditional_spambots_3 = pd.read_csv('datasets\\cresci-2017.csv\\datasets_full.csv\\traditional_spambots_3.csv\\traditional_spambots_3.csv\\users.csv')

traditional_spambots_4 = pd.read_csv('datasets\\cresci-2017.csv\\datasets_full.csv\\traditional_spambots_4.csv\\traditional_spambots_4.csv\\users.csv')


frames = [traditional_spambots_1 , traditional_spambots_2, traditional_spambots_3 , traditional_spambots_4]

traditional_spambots = pd.concat(frames)

frames = [traditional_spambots, social_spambots]

bots = pd.concat(frames)

bots['label'] = 1

frames = (users, bots)

dataset = pd.concat(frames)
print(dataset)

dataset.to_csv("dataset.csv", index=False)