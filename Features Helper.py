import csv
import os
import time
from pathlib import Path
import dateutil
from dateutil import parser


def behaviouralFutures():


    flag = True
    a = None
    counter = 0
    average = 0
    tweetTimeTotal = 0
    totalTTSD = 0
    totalTTISD = 0
    timesArray = []
    timesIntervalArray = []

    for row in dictionary:
        counter = counter + 1
        date = dateutil.parser.parse(row['created_at'])
        if flag:
            a = time.strptime(str(date), '%Y-%m-%d %H:%M:%S+00:00')
            a = time.mktime(a)
            timesArray.append(a)
            tweetTimeTotal = tweetTimeTotal + a
            flag = False
        else:
            b = time.strptime(str(date), '%Y-%m-%d %H:%M:%S+00:00')
            b = time.mktime(b)
            timesArray.append(b)
            tweetTimeTotal = tweetTimeTotal + b
            d = a - b
            difference = int(d) / 60 / 60
            timesIntervalArray.append(difference)
            average = average + difference
            a = b
    average = average / counter  # Average Time between tweets
    tweetTimeAverage = tweetTimeTotal / counter
    for elem in timesArray:
        totalTTSD = totalTTSD + ((elem - tweetTimeAverage) / 86400) ** 2
    TTSD = totalTTSD / counter  # Tweet Time Standard Deviation
    for elem in timesIntervalArray:
        totalTTISD = totalTTISD + (elem - average) ** 2
    TTISD = totalTTISD / counter

    print(average)  # Average Time Between Tweets Given in Hours
    print(TTSD)  # Tweet Time Standard Deviation Given in Days
    print(TTISD)  # Tweet Time Interval Standard Deviation Given in Hours


def contentBasedFutures():
    dictionary = csv.DictReader(open(str(Path(os.getcwd())) + "/Tweets.csv", mode='r'))

    retweetCount = 0
    totalLikes = 0
    totalHashtags = 0
    totalRetweets = 0
    totalURLs = 0
    totalMentions = 0
    counter = 0

    for row in dictionary:
        counter = counter + 1
        tweetText = row['text']
        publicMetricsString = row['public_metrics']
        if tweetText[0:2] == 'RT':
            retweetCount = retweetCount + 1

        indexStartLike = publicMetricsString.find('like_count') + 13
        indexStartRetweet = publicMetricsString.find('retweet_count') + 16
        totalLikes = totalLikes + int(publicMetricsString[indexStartLike:len(publicMetricsString)].split(',')[0])
        totalRetweets = totalRetweets + int(row['public_metrics'][indexStartRetweet:len(publicMetricsString)].split(',')[0])
        totalHashtags = totalHashtags + tweetText.count('#')
        totalMentions = totalMentions + tweetText.count('@')
        if row['referenced_tweets'] == "":
            totalURLs = totalURLs + tweetText.count('https://') + tweetText.count('http://')
        else:
            if row['referenced_tweets'].find('quoted') == -1:
                totalURLs = totalURLs + tweetText.count('https://') + tweetText.count('http://')
            else:
                if tweetText.count('https://') + tweetText.count('http://') > 1:
                    totalURLs = totalURLs + tweetText.count('https://') + tweetText.count('http://') - 1

        # print(row['public_metrics'])

    retweetRatio = retweetCount / counter
    averageLikes = totalLikes / counter
    averageHashtags = totalHashtags / counter
    averageRetweets = totalRetweets / counter
    averageURLs = totalURLs / counter
    averageMentions = totalMentions / counter

    print(retweetRatio)  # Retweet Ratio
    print(averageLikes)  # Average Likes Per Tweet
    print(averageHashtags)  # Average Hashtags Per Tweet
    print(averageRetweets)  # Average Retweets Per Tweet
    print(averageURLs)  # Average URLs Per Tweet
    print(averageMentions)  # Average Mentions Per Tweet



# behaviouralFutures()
contentBasedFutures()
