from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
from sklearn.metrics import confusion_matrix as cm, classification_report
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as metrics
import itertools
from sklearn.metrics import accuracy_score
from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline


crime_data = pd.read_csv('data/crime_data_preprocessed.csv', engine='python')
crime_data = crime_data.head(20000)


def balance_classes(xs, ys):
    ##"""Undersample xs, ys to balance classes."""
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


print(Counter(crime_data['class_value']))
balanced_x, balanced_y = balance_classes(
    crime_data['original_text'], crime_data['class_value'])
print(Counter(balanced_y))


Train_X_balanced, Test_X_balanced, Train_Y_balanced, Test_Y_balanced = model_selection.train_test_split(
    crime_data['original_text'], crime_data['class_value'], test_size=0.3)


vec_transformed = CountVectorizer(min_df=200, max_df=150.0, lowercase=True)
data = vec_transformed.fit_transform(crime_data['original_text'])


tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(data)


# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_,
                      index=vec_transformed.get_feature_names(), columns=["idf_weights"])

# sort ascending
df_idf.sort_values(by=['idf_weights'])
# print(df_idf)

# count matrix
count_vector = vec_transformed.transform(crime_data['original_text'])

# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(count_vector)
tf_idf_vector.shape
print(tf_idf_vector.shape)


tf_idf_vector.shape
feature_names = vec_transformed.get_feature_names()

# get tfidf vector for first document
first_document_vector = tf_idf_vector[0]

# print the scores
df = pd.DataFrame(first_document_vector.T.todense(),
                  index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"], ascending=False)
# print(df)


# count matrix
count_vector = vec_transformed.transform(Train_X_balanced)

# tf-idf scores
Train_X_Tfidf = tfidf_transformer.transform(count_vector)

# count matrix
count_vector = vec_transformed.transform(Test_X_balanced)

# tf-idf scores
Test_X_Tfidf = tfidf_transformer.transform(count_vector)


# count matrix
count_vector = vec_transformed.transform(Test_X_balanced)

# tf-idf scores
Test_X_Tfidf = tfidf_transformer.transform(count_vector)

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(Train_X_Tfidf, Train_Y_balanced)

y_pred = logreg.predict(Test_X_Tfidf)
# print(cm(Test_Y_balanced, y_pred))
# print('\n')
# print("Classification Report ")
# print(classification_report(Test_Y_balanced, y_pred))
# print("log reg  Accuracy Score ->",
#       metrics.accuracy_score(Test_Y_balanced, y_pred))


model = LinearSVC()
model.fit(Train_X_Tfidf, Train_Y_balanced)
y_pred_svc1 = model.predict(Test_X_Tfidf)
# print(cm(Test_Y_balanced, y_pred_svc1))
# print('\n')
# print("Classification Report ")
# print(classification_report(Test_Y_balanced, y_pred_svc1))
# print("log reg  Accuracy Score ->",
#       metrics.accuracy_score(Test_Y_balanced, y_pred_svc1))


vec_transformed = CountVectorizer(
    min_df=200, max_df=150.0, lowercase=True, ngram_range=(1, 2), analyzer='word')
data = vec_transformed.fit_transform(crime_data['original_text'])

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(data)

df_idf = pd.DataFrame(tfidf_transformer.idf_,
                      index=vec_transformed.get_feature_names(), columns=["idf_weights"])

# sort ascending
df_idf.sort_values(by=['idf_weights'])
# print(df_idf)


# count matrix
count_vector = vec_transformed.transform(crime_data['original_text'])

# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(count_vector)
# print(tf_idf_vector)

feature_names = vec_transformed.get_feature_names()

# get tfidf vector for first document
first_document_vector = tf_idf_vector[0]

# print the scores
df = pd.DataFrame(first_document_vector.T.todense(),
                  index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"], ascending=False)

# print(df)

count_vector = vec_transformed.transform(Train_X_balanced)

# tf-idf scores
Train_X_Tfidf = tfidf_transformer.transform(count_vector)


# count matrix
count_vector = vec_transformed.transform(Test_X_balanced)

# tf-idf scores
Test_X_Tfidf = tfidf_transformer.transform(count_vector)


logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(Train_X_Tfidf, Train_Y_balanced)
y_pred = logreg.predict(Test_X_Tfidf)
# print(cm(Test_Y_balanced, y_pred))
# print('\n')
# print("Classification Report ")
# print(classification_report(Test_Y_balanced, y_pred))
# print("log reg  Accuracy Score ->",
#       metrics.accuracy_score(Test_Y_balanced, y_pred))


gb_clf = GradientBoostingClassifier(
    n_estimators=20, learning_rate=1.0, max_features=200, max_depth=50, random_state=0)
gb_clf.fit(Train_X_Tfidf, Train_Y_balanced)

y_pred = gb_clf.predict(Test_X_Tfidf)
# print(cm(Test_Y_balanced, y_pred))
# print('\n')
# print("Classification Report ")
# print(classification_report(Test_Y_balanced, y_pred))
# print("grad Accuracy Score ->", metrics.accuracy_score(Test_Y_balanced, y_pred))


vec_transformed = CountVectorizer(
    min_df=400, max_df=290.0, lowercase=True, ngram_range=(1, 3), analyzer='word')
data = vec_transformed.fit_transform(crime_data['original_text'])

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(data)

df_idf = pd.DataFrame(tfidf_transformer.idf_,
                      index=vec_transformed.get_feature_names(), columns=["idf_weights"])

# sort ascending
df_idf.sort_values(by=['idf_weights'])
# print(df_idf)


# count matrix
count_vector = vec_transformed.transform(crime_data['original_text'])

# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(count_vector)
tf_idf_vector.shape

print(tf_idf_vector.shape)


feature_names = vec_transformed.get_feature_names()

# get tfidf vector for first document
first_document_vector = tf_idf_vector[0]

# print the scores
df = pd.DataFrame(first_document_vector.T.todense(),
                  index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"], ascending=False)


# count matrix
count_vector = vec_transformed.transform(Train_X_balanced)

# tf-idf scores
Train_X_Tfidf = tfidf_transformer.transform(count_vector)

# count matrix
count_vector = vec_transformed.transform(Test_X_balanced)

# tf-idf scores
Test_X_Tfidf = tfidf_transformer.transform(count_vector)


model = LinearSVC()
model.fit(Train_X_Tfidf, Train_Y_balanced)
y_pred_svc1 = model.predict(Test_X_Tfidf)
print(cm(Test_Y_balanced, y_pred_svc1))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y_balanced, y_pred_svc1))
print("log reg  Accuracy Score ->",
      metrics.accuracy_score(Test_Y_balanced, y_pred_svc1))


model = LinearSVC()
model.fit(Train_X_Tfidf, Train_Y_balanced)
y_pred_svc1 = model.predict(Test_X_Tfidf)
print(cm(Test_Y_balanced, y_pred_svc1))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y_balanced, y_pred_svc1))
print("log reg  Accuracy Score ->",
      metrics.accuracy_score(Test_Y_balanced, y_pred_svc1))


logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(Train_X_Tfidf, Train_Y_balanced)
y_pred = logreg.predict(Test_X_Tfidf)
print(cm(Test_Y_balanced, y_pred))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y_balanced, y_pred))
print("log reg  Accuracy Score ->",
      metrics.accuracy_score(Test_Y_balanced, y_pred))


vec_transformed = CountVectorizer(
    min_df=400, max_df=150.0, lowercase=True, ngram_range=(1, 4), analyzer='word')
data = vec_transformed.fit_transform(crime_data['original_text'])

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(data)

df_idf = pd.DataFrame(tfidf_transformer.idf_,
                      index=vec_transformed.get_feature_names(), columns=["idf_weights"])

# sort ascending
df_idf.sort_values(by=['idf_weights'])
print(df_idf)


count_vector = vec_transformed.transform(crime_data['original_text'])

# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(count_vector)

# count matrix
count_vector = vec_transformed.transform(Train_X_balanced)

# tf-idf scores
Train_X_Tfidf = tfidf_transformer.transform(count_vector)

# count matrix
count_vector = vec_transformed.transform(Test_X_balanced)

# tf-idf scores
Test_X_Tfidf = tfidf_transformer.transform(count_vector)


model = LinearSVC()
model.fit(Train_X_Tfidf, Train_Y_balanced)
y_pred_svc1 = model.predict(Test_X_Tfidf)
print(cm(Test_Y_balanced, y_pred_svc1))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y_balanced, y_pred_svc1))
print("log reg  Accuracy Score ->",
      metrics.accuracy_score(Test_Y_balanced, y_pred_svc1))

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(Train_X_Tfidf, Train_Y_balanced)
y_pred = logreg.predict(Test_X_Tfidf)
print(cm(Test_Y_balanced, y_pred))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y_balanced, y_pred))
print("log reg  Accuracy Score ->",
      metrics.accuracy_score(Test_Y_balanced, y_pred))


gb_clf = GradientBoostingClassifier(
    n_estimators=20, learning_rate=1.0, max_features=100, max_depth=50, random_state=0)
gb_clf.fit(Train_X_Tfidf, Train_Y_balanced)

y_pred = gb_clf.predict(Test_X_Tfidf)
print(cm(Test_Y_balanced, y_pred))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y_balanced, y_pred))
print("grad Accuracy Score ->", metrics.accuracy_score(Test_Y_balanced, y_pred))
