
import warnings
import sys
import re
from nltk.stem.snowball import SnowballStemmer
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

crime_data = pd.read_csv('data/crime_data_preprocessed.csv', engine='python')

np.random.seed(500)
crime_data['original_text'].dropna(inplace=True)
crime_data['original_text'] = [entry.lower()
                               for entry in crime_data['original_text']]


if not sys.warnoptions:
    warnings.simplefilter("ignore")


def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def cleanPunc(sentence):  # function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


crime_data['clean_text'] = crime_data['original_text'].str.lower()
crime_data['clean_text'] = crime_data['clean_text'].apply(cleanHtml)
crime_data['clean_text'] = crime_data['clean_text'].apply(cleanPunc)
crime_data['clean_text'] = crime_data['clean_text'].apply(keepAlpha)
crime_data.head()

stop_words = set(stopwords.words('english'))

re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)


def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)


crime_data['clean_text'] = crime_data['original_text'].apply(removeStopWords)
crime_data.head()

stemmer = SnowballStemmer("english")


def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


# crime_data['clean_text'] = crime_data['clean_text'].apply(stemming)
# crime_data.head()
# print(crime_data.head())


# Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(
#     crime_data['clean_text'], crime_data['class_value'], test_size=0.3)

# Encoder = LabelEncoder()
# Train_Y = Encoder.fit_transform(Train_Y)
# Test_Y = Encoder.fit_transform(Test_Y)


# Tfidf_vect = TfidfVectorizer(max_features=5000)
# Tfidf_vect.fit(crime_data['clean_text'])

# Train_X_Tfidf = Tfidf_vect.transform(Train_X)
# Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# print(Train_X_Tfidf)

# # print(Tfidf_vect.vocabulary_)


# # fit the training dataset on the NB classifier
# Naive = naive_bayes.MultinomialNB()
# Naive.fit(Train_X_Tfidf, Train_Y)
# # predict the labels on validation dataset
# predictions_NB = Naive.predict(Test_X_Tfidf)
# # Use accuracy_score function to get the accuracy
# print("Naive Bayes Accuracy Score -> ",
#       accuracy_score(predictions_NB, Test_Y)*100)


# # Classifier - Algorithm - SVM
# # fit the training dataset on the classifier
# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
# SVM.fit(Train_X_Tfidf, Train_Y)
# # predict the labels on validation dataset
# predictions_SVM = SVM.predict(Test_X_Tfidf)
# # Use accuracy_score function to get the accuracy
# print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y)*100)


# # export_csv = crime_data.to_csv("./data/crime_data_preprocessed.csv",
#                             #    index=None, header=True)  # Don't forget to add '.csv' at the end of the path
