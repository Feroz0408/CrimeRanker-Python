from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
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
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

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


np.random.seed(500)
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(
    crime_data['original_text'], crime_data['class_value'], test_size=0.3)


Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


Tfidf_vect = TfidfVectorizer(max_df=100.0, min_df=200)
Tfidf_vect.fit(crime_data['original_text'].astype('U').values)
print(Tfidf_vect.vocabulary_)


Train_X_Tfidf = Tfidf_vect.transform(Train_X.values.astype('U'))
Test_X_Tfidf = Tfidf_vect.transform(Test_X.values.astype('U'))


gb_clf = GradientBoostingClassifier(
    n_estimators=20, learning_rate=0.75, max_features=200, max_depth=55, random_state=1)
gb_clf.fit(Train_X_Tfidf, Train_Y)


# Use accuracy_score function to get the accuracy
# print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)


predictions = gb_clf.predict(Test_X_Tfidf)
print(cm(Test_Y, predictions))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y, predictions.astype(np.int)))
print("Random forest Accuracy Score ->",
      metrics.accuracy_score(Test_Y, predictions.astype(np.int)))


# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
# print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
print(cm(Test_Y, predictions_NB))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y, predictions_NB))
print("Naive Bayes Accuracy Score ->",
      metrics.accuracy_score(Test_Y, predictions_NB))


logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(Train_X_Tfidf, Train_Y)
y_pred = logreg.predict(Test_X_Tfidf)
print(cm(Test_Y, y_pred))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y, y_pred))
print("log reg  Accuracy Score ->", metrics.accuracy_score(Test_Y, y_pred))


model = LinearSVC()
model.fit(Train_X_Tfidf, Train_Y)
print('----')
y_pred_svc = model.predict(Test_X_Tfidf)
print(cm(Test_Y, y_pred_svc))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y, y_pred_svc))
print("linear SVC Accuracy Score  Count vect->",
      metrics.accuracy_score(Test_Y, y_pred_svc))


vec = CountVectorizer()
data = vec.fit_transform(crime_data['original_text'])


vector = vec.transform(crime_data['original_text'])
# summarize encoded vector
print(vector.shape)
print(type(vector))


Train_X_count_vec = vec.transform(Train_X.values.astype('U'))
Test_X_count_vec = vec.transform(Test_X.values.astype('U'))


gb_clf = GradientBoostingClassifier(
    n_estimators=20, learning_rate=0.5, max_features=100, max_depth=55, random_state=0)
gb_clf.fit(Train_X_count_vec, Train_Y)


predictions = gb_clf.predict(Test_X_count_vec)
print(cm(Test_Y, predictions))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y, predictions))
print("gradiant forest Accuracy Score ->",
      metrics.accuracy_score(Test_Y, predictions))


Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_count_vec, Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_count_vec)
# Use accuracy_score function to get the accuracy
# print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
print(cm(Test_Y, predictions_NB))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y, predictions_NB))
print("Naive Bayes Accuracy Score using count vect ->",
      metrics.accuracy_score(Test_Y, predictions_NB))


logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(Train_X_count_vec, Train_Y)
y_pred = logreg.predict(Test_X_count_vec)
print(cm(Test_Y, y_pred))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y, y_pred))
print("log reg  Accuracy Score ->", metrics.accuracy_score(Test_Y, y_pred))


model = LinearSVC()
model.fit(Train_X_count_vec, Train_Y)
print('----')
y_pred_svc = model.predict(Test_X_count_vec)
print(cm(Test_Y, y_pred_svc))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y, y_pred_svc))
print("linear SVC Accuracy Score  Count vect->",
      metrics.accuracy_score(Test_Y, y_pred_svc))


vec_transformed = CountVectorizer(min_df=200, max_df=150.0, lowercase=True)
data = vec_transformed.fit_transform(crime_data['original_text'])


Train_X_count_vec_tran = vec_transformed.transform(Train_X.values.astype('U'))
Test_X_count_vec_tran = vec_transformed.transform(Test_X.values.astype('U'))


Naive_count = naive_bayes.MultinomialNB()
Naive_count.fit(Train_X_count_vec_tran, Train_Y)
# predict the labels on validation dataset
predictions_NB_vec = Naive_count.predict(Test_X_count_vec_tran)
# Use accuracy_score function to get the accuracy
#print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
print(cm(Test_Y, predictions_NB_vec))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y, predictions_NB_vec))
print("Naive Bayes Accuracy Score using count vect min df ->",
      metrics.accuracy_score(Test_Y, predictions_NB_vec))

model = LinearSVC()
model.fit(Train_X_count_vec_tran, Train_Y)
y_pred_svc1 = model.predict(Test_X_count_vec_tran)
print(cm(Test_Y, y_pred_svc1))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y, y_pred_svc1))
print("linear SVC Accuracy Score  Count vect->",
      metrics.accuracy_score(Test_Y, y_pred_svc1))

logreg1 = LogisticRegression(n_jobs=1, C=1e5)
logreg1.fit(Train_X_count_vec_tran, Train_Y)
y_pred_vect_1 = logreg1.predict(Test_X_count_vec_tran)
print(cm(Test_Y, y_pred_vect_1))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y, y_pred_vect_1))
print("log reg  Accuracy Score ->", metrics.accuracy_score(Test_Y, y_pred_vect_1))

kmodel = KNeighborsClassifier(n_neighbors=3)
kmodel.fit(Train_X_count_vec_tran, Train_Y)
y_pred_vect_1 = kmodel.predict(Test_X_count_vec_tran)
print(cm(Test_Y, y_pred_vect_1))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y, y_pred_vect_1))
print("knn  Accuracy Score ->", metrics.accuracy_score(Test_Y, y_pred_vect_1))

balanced_x, balanced_y = balance_classes(
    crime_data['original_text'], crime_data['class_value'])
print(Counter(balanced_y))

Train_X_balanced, Test_X_balanced, Train_Y_balanced, Test_Y_balanced = model_selection.train_test_split(
    balanced_x, balanced_y, test_size=0.3)


vec_transformed = CountVectorizer(min_df=200, max_df=150.0, lowercase=True)
data = vec_transformed.fit_transform(balanced_x)


Train_X_count_vec_bal = vec_transformed.transform(Train_X_balanced)
Test_X_count_vec_bal = vec_transformed.transform(Test_X_balanced)



Naive_count = naive_bayes.MultinomialNB()
Naive_count.fit(Train_X_count_vec_bal,Train_Y_balanced)
# predict the labels on validation dataset
predictions_NB_vec = Naive_count.predict(Test_X_count_vec_bal)
# Use accuracy_score function to get the accuracy
#print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
print(cm(Test_Y_balanced, predictions_NB_vec))
print('\n')
print("Classification Report ")
print(classification_report(Test_Y_balanced, predictions_NB_vec))
print("Naive Bayes Accuracy Score using count vect min df ->",metrics.accuracy_score(Test_Y_balanced, predictions_NB_vec))

