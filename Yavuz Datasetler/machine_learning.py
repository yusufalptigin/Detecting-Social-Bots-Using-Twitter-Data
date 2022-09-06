import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from Levenshtein import distance as lev

def string_length(x):   # returns string length
    return len(x)


def string_digit_count(sstring):    #returns string digit count
    count = 0
    for character in sstring:
        if character.isdigit():
            count += 1
    return count

def lev_distance(username, screenname):

    return lev(username,screenname)



dataset = pd.read_csv('../dataset.csv')

account_features_dataset = dataset[["name","screen_name","default_profile","geo_enabled","protected","verified","statuses_count","followers_count", "friends_count", "favourites_count", "listed_count"]]

account_features_dataset["name"] = account_features_dataset["name"].astype('string')

account_features_dataset["screen_name"] = account_features_dataset["screen_name"].astype('string')

print(account_features_dataset.info()) # shows dtype and non null values

account_features_dataset[['default_profile', 'geo_enabled', 'protected', 'verified']] = account_features_dataset[['default_profile', 'geo_enabled', 'protected', 'verified']].fillna(value=0)

account_features_dataset.dropna()

print(account_features_dataset.info()) # na filled with 0

account_features_dataset['user_name_length'] = account_features_dataset["name"].apply(string_length)

account_features_dataset['screen_name_length'] = account_features_dataset["screen_name"].apply(string_length)

account_features_dataset['screen_name_digits'] = account_features_dataset["screen_name"].apply(string_digit_count)

print(account_features_dataset.head())

account_features_dataset['lev_distance'] = account_features_dataset.apply(lambda x: lev_distance(x["name"], x["screen_name"]), axis=1)

account_features_dataset = account_features_dataset.drop(["name","screen_name"], axis=1)





label_set = dataset["label"]

X_train, X_test, y_train, y_test = train_test_split(account_features_dataset, label_set, test_size=0.2)

rf = RandomForestClassifier(n_estimators=150)
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))
prediction = rf.predict(X_test)
print(accuracy_score(y_test, prediction))
print(confusion_matrix(y_test, prediction))

