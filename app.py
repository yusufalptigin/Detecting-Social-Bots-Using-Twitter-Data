from Levenshtein import distance as lev
from datetime import *
import pandas as pd
import emoji
import re
import numpy as np
#Import Flask modules
from flask import Flask, request, render_template

#Import pickle to save our model
import pickle 

#twitter libraries
import tweepy
import config

    
def string_digit_count(sstring):    #returns string digit count
    count = 0
    for character in sstring:
        if character.isdigit():
            count += 1
    return count

def find_number_of_hashtags(text):
    words = text.split(' ')
    count = 0
    for word in words:
        if word.startswith('#'):
            count += 1

    return count

def count_emoji(text):
    text =  emoji.demojize(text)
    result = re.findall(r'(:[!_\-\w]+:)', text)
    return len(result)


def extract_features(df):
    tweet_time_array = []
    tweet_time_difference = 0
    retweet_count = 0
    retweet_time_array = []
    retweet_time_difference = 0
    tweet_count = 0
    emoji_count = 0
    tweet_size_count = 0
    number_of_hashtags = 0
    number_of_urls = 0

    for tweet in tweets:
        if type(tweet.text) == str and (tweet.text.startswith("RT @")): 
            retweet_count += 1
            retweet_time_array.append(tweet.created_at)

        else:
            tweet_count += 1
            tweet_time_array.append(tweet.created_at)
            text = tweet.text
            if type(text) != str:
                text = str(text)
            tweet_size_count += len(text)
            number_of_hashtags += len(tweet.entities["hashtags"])
            emoji_count += count_emoji(text)
            number_of_urls += len(tweet.entities["urls"])


    for x in range(tweet_count-1):
        time1 = tweet_time_array[x+1]
        time2 = tweet_time_array[x]      
        
        diff = time2 - time1
        tweet_time_difference += diff.total_seconds() / 3600


    for x in range(retweet_count-1):
        time1 = retweet_time_array[x+1]
        time2 = retweet_time_array[x]     
        
        diff = time2 - time1
        retweet_time_difference += diff.total_seconds() / 3600

    df["tweet_rate"] = 0 if tweet_count < 2 else tweet_time_difference / tweet_count 
    df["retweet_rate"] = 0 if retweet_count < 2 else retweet_time_difference / retweet_count  
    df["emoji_count"] = emoji_count / tweet_count if tweet_count > 0 else 0
    df["tweet_size"] = tweet_size_count / tweet_count if tweet_count > 0 else 0
    df["number_of_hashtags"] = number_of_hashtags / tweet_count if tweet_count > 0 else 0
    df["number_of_urls"] = number_of_urls / tweet_count if tweet_count > 0 else 0
    return df


auth = tweepy.OAuth1UserHandler(
   config.API_KEY, config.API_SECRET,
   config.ACCESS_TOKEN, config.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

user = api.get_user(screen_name = "sozluk") 

"""

#feature vector for paper 1 account features
feature_vector = [user.default_profile,user.geo_enabled, user.protected,user.verified,user.statuses_count,
                  user.followers_count,user.friends_count, user.favourites_count,user.listed_count,
                  len(user.name),len(user.screen_name),string_digit_count(user.screen_name),
                  lev(user.screen_name,user.name)]

#feature vector for paper tweet features
tweets = api.user_timeline(id = 1476564123911659524, count=100)

d = {'id': [1476564123911659524]}
dataframe = pd.DataFrame(data=d)
dataframe = dataframe.apply(extract_features, axis=1)
dataframe = dataframe.drop(["id"], axis = 1)
feature_vector = [dataframe["tweet_rate"], dataframe["retweet_rate"],dataframe["emoji_count"],dataframe["tweet_size"],
                  dataframe["number_of_hashtags"],dataframe["number_of_urls"]]

"""


#feature vector for paper 1 all features
tweets = api.user_timeline(screen_name = "sozluk", count=100)

d = {'id': [user.id]} 
dataframe = pd.DataFrame(data=d)
dataframe = dataframe.apply(extract_features, axis=1)
dataframe = dataframe.drop(["id"], axis = 1)




feature_vector = [dataframe["tweet_rate"], dataframe["retweet_rate"],dataframe["emoji_count"],dataframe["tweet_size"],
                  dataframe["number_of_hashtags"],dataframe["number_of_urls"],
                  user.default_profile,user.geo_enabled, user.protected,user.verified,user.statuses_count,
                  user.followers_count,user.friends_count, user.favourites_count,user.listed_count,
                  len(user.name),len(user.screen_name),string_digit_count(user.screen_name),
                  lev(user.screen_name,user.name)]
                  
                  


model = pickle.load(open('model.pkl','rb'))

feature_vector = [np.array(feature_vector)]

prediction = model.predict(feature_vector)

print(prediction)


"""

#Initialize Flask and set the template folder to "template"
app = Flask(__name__, template_folder = 'template')

#Open our model 
model = pickle.load(open('model.pkl','rb'))

#create our "home" route using the "index.html" page
@app.route('/')
def home():
    return render_template('index.html')

#Set a post method to yield predictions on page
@app.route('/', methods = ['POST'])
def predict():
    
    #obtain all form values and place them in an array, convert into integers
    Account_name = request.form['account_name']

    #predict the price given the values inputted by user
    #prediction = model.predict(final_features)

 
    #Round the output to 2 decimal places
    #output = round(prediction[0], 2)
    output = 1 
    #If the output is negative, the values entered are unreasonable to the context of the application
    #If the output is greater than 0, return prediction
    if output < 0:
        return render_template('index.html', prediction_text = "Predicted Price is negative, values entered not reasonable")
    elif output >= 0:
        return render_template('index.html', prediction_text = 'Predicted Price of the house is: ${}'.format(output))   

#Run app
if __name__ == "__main__":
    app.run(debug=True)
    """