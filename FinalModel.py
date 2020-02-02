from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as cm, classification_report
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

crime_data = pd.read_csv('data/crime_data_preprocessed.csv', engine='python')
crime_data = crime_data.head(20000)

live_crime_data = pd.read_csv(
    'data/live_data_preprocessed.csv', engine='python')
live_crime_data = live_crime_data.head(500)


def class_balance(xs, ys):
    # Under sampling to equally distribute classes for accurate prediction
    freqs = Counter(ys)
    max_allowable = freqs.most_common()[-1][1]
    num_added = {clss: 0 for clss in freqs.keys()}
    new_ys = []
    new_xs = []
    for i, y in enumerate(ys):
        if num_added[y] < max_allowable:
            new_ys.append(y)
            new_xs.append(xs[i])
            num_added[y] += 1
    return new_xs, new_ys


balance_x, balance_y = class_balance(
    crime_data['original_text'], crime_data['class_value'])

# print("Class Balancing")
# print("======================")
# print(Counter(balance_y))
# print("======================")

np.random.seed(500)
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(
    crime_data['original_text'], crime_data['class_value'], test_size=0.3)

Train_X1, Test_X1, Train_Y1, Test_Y1 = model_selection.train_test_split(
    live_crime_data['text'], live_crime_data['text'], test_size=0.8)


Label_Encoder = LabelEncoder()
Train_Y = Label_Encoder.fit_transform(Train_Y)
Test_Y = Label_Encoder.fit_transform(Test_Y)


Label_Encoder = LabelEncoder()
Train_Y1 = Label_Encoder.fit_transform(Train_Y1)
Test_Y1 = Label_Encoder.fit_transform(Test_Y1)


Vec_Count = CountVectorizer()
data = Vec_Count.fit_transform(crime_data['original_text'])
vector = Vec_Count.transform(crime_data['original_text'])
# Summary of encoded vectors
# print(vector.shape)

Train_X_count_vec = Vec_Count.transform(Train_X.values.astype('U'))
Test_X_count_vec = Vec_Count.transform(Test_X.values.astype('U'))


Train_X1_count_vec = Vec_Count.transform(Train_X1.values.astype('U'))
Test_X1_count_vec = Vec_Count.transform(Test_X1.values.astype('U'))


Logistic_Reg_Model = LogisticRegression(n_jobs=1, C=1e5)
Logistic_Reg_Model.fit(Train_X_count_vec, Train_Y)

# Predicting class labels for live tweets fetched from twitter
Class_Predictions = Logistic_Reg_Model.predict(Test_X1_count_vec)
live_data_classes = Class_Predictions
# print(Class_Predictions)
# print("Confusion Matrix")
# print(cm(Test_Y, Class_Predictions))
# print('\n')
# print("Classification Report ")
# print(classification_report(Test_Y1, Class_Predictions))
# print("Logistic Regression Accuracy Score Count Vect ->",
#       metrics.accuracy_score(Test_Y1, Class_Predictions))


# Logistic_Reg_Model = LogisticRegression(n_jobs=1, C=1e5)
# Logistic_Reg_Model.fit(Train_X_count_vec, Train_Y)
# Class_Predictions = Logistic_Reg_Model.predict(Test_X_count_vec)
# print("Confusion Matrix")
# print(cm(Test_Y, Class_Predictions))
# print('\n')
# print("Classification Report ")
# print(classification_report(Test_Y, Class_Predictions))
# print("Logistic Regression Accuracy Score Count_Vectorization ->",
#       metrics.accuracy_score(Test_Y, Class_Predictions))
