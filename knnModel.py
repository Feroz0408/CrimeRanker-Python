import pandas as pd
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import nltk
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split


def preprocess_text(train):
    train_list1 = train['original_text'].values.tolist()
    lb = LabelEncoder()
    train['original_text'] = lb.fit_transform(train_list1)
    return(train)

# Below module Calculates the accuracy of prediction by K-nn algorithm


def accuracyKnn(testresult, test):
    # correct = (len(test)-1800)
    cnt = 0
    # correct = len(test)
    Label = ['class_value']
    testLabel = test.as_matrix(Label)
    for i in range(len(testresult)):
        if testLabel[i][0] == testresult[i]:
            cnt += 1
    print("\nK-Neighbor Classifier Results:")
    print("--------------------------------------------")
    print("Total Docs Correctly Classified=", cnt)
    print("Total Number of Test Documents=", len(
        testresult))
    print
    accuracy = (cnt / len(testresult))
    # Classification_report(testLabel,testresult,0)
    return accuracy


# Below function performs KNN classification on tarining data and returns the output of prediction performed on Test data

def knn(train, test):
    cols = ['original_text']
    Label = ['class_value']
    trainData = train.as_matrix(cols)  # x-train
    trainLabel = train.as_matrix(Label)  # y-train
    testData = test.as_matrix(cols)  # x-validation
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(trainData, trainLabel.ravel())
    output = knn.predict(testData)
    return output


if __name__ == '__main__':
    # K-n Classifier
    train = pd.read_csv('./data/crime_data.csv', engine='python',
                        usecols=['original_text', 'class_value'])
    test = pd.read_csv('./data/crime_test.csv', engine='python',
                       usecols=['original_text', 'class_value'])

    train = preprocess_text(train)
    test = preprocess_text(test)

    output = knn(train, test)

    accuracy = accuracyKnn(output, test)
    print("Accuracy with k-neighbor=", accuracy)

    # df['class_value'] = df['hashtags'].apply(lambda x: 0 if x ==’Epistemology’ else 1)
    # X_train, X_test, y_train, y_test = train_test_split(df['hashtags’], df['class_value'], random_state=1)
