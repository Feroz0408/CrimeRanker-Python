import tweepy
import sys
import re
import time
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import csv
import pandas as pd
import Stemming_Preprocessing
import os
from os import path
import CrimeModels

xmin = -92.888114
ymin = 42.491983
xmax = -86.805415
ymax = 47.080621

if path.exists("twitDB.json"):
    os.remove("twitDB.json")

# authorization tokens
consumer_key = 'rmd6kFz3ej0RlgNRMJM1abZ6C'
consumer_secret = 'xWFHlAgDysUHIsqcZIR5p6T3b5xMlzs5FIQRnl3fhjFcEXI4PX'
access_key = '962442942093852673-Fm8zMecozk0mfrbZ7FKgT4D1YYdVz5h'
access_secret = 'ExmG8uAv1X0t0iBSrDbyymej5vFdUXoYRgg7K3ZRuSN9k'


class listener(StreamListener):

    def __init__(self):
        super().__init__()
        self.counter = 0
        self.limit = 200

    def on_data(self, data):
        try:
            saveFile = open('twitDB.json', 'a')
            saveFile.write(data)
            saveFile.close()

            self.counter += 1
            if self.counter < self.limit:
                return True
            else:
                twitterStream.disconnect()

        except BaseException as e:
            print('Failed', str(e))
            time.sleep(5)

    def on_error(self, status):
        print(status)


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
tags = ["molested", "rape", "statutoryrape", "sexualvoilence", "molest", "gangrape", "girlabuse", "assault", "childabuse", "abuse", "domesticabuse", "kidnapping", "burglary",
        "larceny", "robbery", "autotheft", "shoplifting", "theft", "murder", "killed", "drugcrime", "trafficoffense", "financialcrime", "fraud", "drugtrafficking", "blackmail"]


twitterStream = Stream(auth, listener())
twitterStream.filter(track=tags)
# Extracting live tweets based on location
twitterStream.filter(locations=[xmin, ymin, xmax, ymax])

with open('twitDB.json', encoding='utf-8-sig') as f_input:
    df = pd.read_json(f_input,  lines=True)

df.to_csv('twitDB.csv', encoding='utf-8', index=False)

live_crime_data = pd.read_csv('twitDB.csv', engine='python')
live_crime_data['text'].dropna(inplace=True)
live_crime_data['text'] = [entry.lower()
                           for entry in live_crime_data['text']]


# Pre processing and stemming live data
live_crime_data['clean_text'] = live_crime_data['text'].str.lower()
live_crime_data['clean_text'] = live_crime_data['text'].apply(
    Stemming_Preprocessing.cleanHtml)
live_crime_data['clean_text'] = live_crime_data['text'].apply(
    Stemming_Preprocessing.cleanPunc)
live_crime_data['clean_text'] = live_crime_data['text'].apply(
    Stemming_Preprocessing.keepAlpha)
live_crime_data.head()


live_crime_data['clean_text'] = live_crime_data['clean_text'].apply(
    Stemming_Preprocessing.stemming)
live_crime_data.head()
print(live_crime_data.head())

export_csv = live_crime_data.to_csv("./data/live_data_preprocessed.csv",
                                    index=None, header=True)


live_classes = CrimeModels.live_data_classes
rape = 0
assault = 0
theft = 0
murder = 0
statuatory = 0
for i in live_classes:
    if(i == 0):
        rape += 1
    if(i == 1):
        theft += 1
    if(i == 2):
        assault += 1
    if(i == 3):
        murder += 1
    if(i == 4):
        statuatory += 1


total = rape + assault + theft + murder + statuatory
print(total)
print("rape", rape, round(100 * float(rape)/float(total)))
print("assault", assault, round(100 * float(assault)/float(total)))
print("theft", theft, round(100 * float(theft)/float(total)))
print("murder", murder, round(100 * float(murder)/float(total)))
print("statuatory", statuatory, round(100 * float(statuatory)/float(total)))

crime_obj = {
    total: total,
    rape: round(100 * float(rape)/float(total)),
    assault: round(100 * float(assault)/float(total)),
    theft: round(100 * float(theft)/float(total)),
    murder: round(100 * float(murder)/float(total)),
    statuatory: round(100 * float(statuatory)/float(total))
}
