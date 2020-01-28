import nltk
import os
import pandas as pd
import tweepy
import re
import string
from textblob import TextBlob
import preprocessor as p
from nltk.tokenize import word_tokenize
import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()


# Twitter credentials for the app
consumer_key = 'rmd6kFz3ej0RlgNRMJM1abZ6C'
consumer_secret = 'xWFHlAgDysUHIsqcZIR5p6T3b5xMlzs5FIQRnl3fhjFcEXI4PX'
access_key = '962442942093852673-Fm8zMecozk0mfrbZ7FKgT4D1YYdVz5h'
access_secret = 'ExmG8uAv1X0t0iBSrDbyymej5vFdUXoYRgg7K3ZRuSN9k'

# pass twitter credentials to tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

# file location changed to "data/telemedicine_data_extraction/" for clearer path
# rape_tweets = "data/rape_data1.csv"
# assault_tweets = "data/assault_data2.csv"
theft_tweets = "data/theft_data2.csv"
# murder_tweets = "data/murder_data1.csv"
# statutory_tweets = "data/statutory_data1.csv"

# columns of the csv file
COLS = ['id', 'created_at', 'source', 'original_text', 'clean_text', 'sentiment', 'polarity', 'subjectivity', 'lang',
        'favorite_count', 'retweet_count', 'original_author', 'possibly_sensitive', 'hashtags',
        'user_mentions', 'place', 'place_coord_boundaries']

# set two date variables for date range
start_date = '2019-10-01'
end_date = '2019-10-31'

# Happy emojis
emojis_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
])

# Sad emojis
emojis_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
])

# Emoji patterns
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emojis
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

# combine sad and happy emojis
emojis = emojis_happy.union(emojis_sad)

# start get stop word


def getStopWordList():
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open('./data/stopwords.txt', 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
# end

# mrhod clean_tweets()


def clean_tweets(tweet):
    # stop_words = set(stopwords.words('english'))
    stop_words = getStopWordList()
    word_tokens = word_tokenize(tweet)

    # after tweepy preprocessing the colon left remain after removing mentions
    # or RT sign in the beginning of the tweet
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    # replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)
    tweet = tweet.lower()
    tweet = re.sub("n't", " not", tweet)
    tweet = re.sub("'s", " is", tweet)
    tweet = re.sub("avg", "average", tweet)
    tweet = re.sub("tnx", "thanks", tweet)
    tweet = re.sub("mkt", "market", tweet)
    tweet = re.sub("nyse", "nice", tweet)
    tweet = re.sub("didn", "did not", tweet)
    tweet = re.sub("don", "do not", tweet)
    tweet = re.sub("min", "minimum", tweet)
    tweet = re.sub("doesn", "does not", tweet)
    tweet = re.sub("dailyfree", "daily free", tweet)
    tweet = re.sub("tech", "technology", tweet)
    tweet = re.sub("bio", "biography", tweet)
    tweet = re.sub("ppl", "people", tweet)
    tweet = re.sub("rpe", "rape", tweet)
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    tweet = re.sub('@[^\s]+', ' ', tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = re.sub('-', '', tweet)
    tweet = re.sub('_', '', tweet)
    tweet = re.sub('\.', '', tweet)
    tweet = re.sub(r'\$.+', '', tweet)
    tweet = re.sub(
        "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet)
    tweet = re.sub("([a-zA-z]+[0-9]+)|([0-9]+)", " ", tweet)
    tweet = re.sub("Ä", " ", tweet)
    tweet = re.sub("¶", " ", tweet)
    tweet = re.sub("Ä¶", " ", tweet)
    tweet = re.sub("ô", " ", tweet)
    tweet = tweet.encode("ascii", errors="ignore").decode()

    # remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)

    # filter using NLTK library append it to a string
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []

    # looping through conditions
    for w in word_tokens:
        # check tokens against stop words , emojis and punctuations
        if w not in stop_words and w not in emojis and w not in string.punctuation:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)
    # print(word_tokens)
    # print(filtered_sentence)

# method write_tweets()


def write_tweets(keyword, file):
    # If the file exists, then read the existing data from the CSV file.
    if os.path.exists(file):
        df = pd.read_csv(file, header=0)
    else:
        df = pd.DataFrame(columns=COLS)
    # page attribute in tweepy.cursor and iteration
    for page in tweepy.Cursor(api.search, q=keyword,
                              count=200, include_rts=False, since=start_date).pages(50):
        for status in page:
            new_entry = []
            status = status._json

            # check whether the tweet is in english or skip to the next tweet
            if status['lang'] != 'en':
                continue

            # when run the code, below code replaces the retweet amount and
            # no of favorires that are changed since last download.
            if status['created_at'] in df['created_at'].values:
                i = df.loc[df['created_at'] == status['created_at']].index[0]
                if status['favorite_count'] != df.at[i, 'favorite_count'] or \
                   status['retweet_count'] != df.at[i, 'retweet_count']:
                    df.at[i, 'favorite_count'] = status['favorite_count']
                    df.at[i, 'retweet_count'] = status['retweet_count']
                continue

           # tweepy preprocessing called for basic preprocessing
            clean_text = p.clean(status['text'])

            # call clean_tweet method for extra preprocessing
            filtered_tweet = clean_tweets(clean_text)

            # pass textBlob method for sentiment calculations
            blob = TextBlob(filtered_tweet)
            Sentiment = blob.sentiment

            # seperate polarity and subjectivity in to two variables
            polarity = Sentiment.polarity
            subjectivity = Sentiment.subjectivity

            # new entry append
            new_entry += [status['id'], status['created_at'],
                          status['source'], status['text'], filtered_tweet, Sentiment, polarity, subjectivity, status['lang'],
                          status['favorite_count'], status['retweet_count']]

            # to append original author of the tweet
            new_entry.append(status['user']['screen_name'])

            try:
                is_sensitive = status['possibly_sensitive']
            except KeyError:
                is_sensitive = None
            new_entry.append(is_sensitive)

            # hashtagas and mentiones are saved using comma separted
            hashtags = ", ".join([hashtag_item['text']
                                  for hashtag_item in status['entities']['hashtags']])
            new_entry.append(hashtags)
            mentions = ", ".join([mention['screen_name']
                                  for mention in status['entities']['user_mentions']])
            new_entry.append(mentions)

            # get location of the tweet if possible
            try:
                location = status['user']['location']
            except TypeError:
                location = ''
            new_entry.append(location)

            try:
                coordinates = [coord for loc in status['place']
                               ['bounding_box']['coordinates'] for coord in loc]
            except TypeError:
                coordinates = None
            new_entry.append(coordinates)

            single_tweet_df = pd.DataFrame([new_entry], columns=COLS)
            df = df.append(single_tweet_df, ignore_index=True)
            csvFile = open(file, 'a', encoding='utf-8')
    df.to_csv(csvFile, mode='a', columns=COLS, index=False, encoding="utf-8")


# declare keywords as a query for three categories
# Rape_keywords = '#molested OR #rape OR #statutoryrape OR #sexualvoilence OR #molest OR #gangrape OR #girlabuse'
# Assault_keywords = '#assault OR #childabuse OR #abuse OR #domesticabuse OR #kidnapping'
Theft_keywords = '#burglary OR #larceny OR #robbery OR #autotheft OR #shoplifting OR #theft'
# Murder_keywords = '#murdered OR #murder OR #killed'
# Statutory_keywords = '#drugcrime OR #trafficoffense OR #financialcrime OR #fraud OR #blackmail OR #drugtrafficking'


# call main method passing keywords and file pat
# write_tweets(Rape_keywords, rape_tweets)
# write_tweets(Assault_keywords,  assault_tweets)
write_tweets(Theft_keywords, theft_tweets)
# write_tweets(Murder_keywords, murder_tweets)
# write_tweets(Statutory_keywords,  statutory_tweets)
