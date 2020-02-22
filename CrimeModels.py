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

import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


crime_data = pd.read_csv('data/crime_data_preprocessed.csv', engine='python')
crime_data = crime_data.head(20000)

live_crime_data = pd.read_csv(
    'data/live_data_preprocessed.csv', engine='python')
live_crime_data = live_crime_data.head(500)


def balance_classes(xs, ys):
    # """Undersample xs, ys to balance classes."""
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


balanced_x, balanced_y = balance_classes(
    crime_data['original_text'], crime_data['class_value'])

# print("Class Balancing")
# print("======================")
# print(Counter(balanced_y))
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

# Tfidf_vect = TfidfVectorizer(max_df=100.0, min_df=200)
# Tfidf_vect.fit(live_crime_data['text'].astype('U').values)

Tfidf_vect = TfidfVectorizer(max_df=1.0, min_df=100)
Tfidf_vect.fit(crime_data['original_text'].astype('U').values)
print("Tfidf_vect.vocabulary_")
print(Tfidf_vect.vocabulary_)


# Train_X_Tfidf = Tfidf_vect.transform(Train_X.values.astype('U'))
# Test_X_Tfidf = Tfidf_vect.transform(Test_X.values.astype('U'))

# Train_X1_Tfidf = Tfidf_vect.transform(Train_X1.values.astype('U'))
# Test_X1_Tfidf = Tfidf_vect.transform(Test_X1.values.astype('U'))


# gb_clf = GradientBoostingClassifier(
#     n_estimators=20, learning_rate=0.75, max_features=200, max_depth=55, random_state=1)
# gb_clf.fit(Train_X_Tfidf, Train_Y)


# Use accuracy_score function to get the accuracy
# print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)


# predictions = gb_clf.predict(Test_X_Tfidf)
# predictions = gb_clf.predict(Test_X1_Tfidf)
# print(predictions)

# print("_____________++++++++++++++++####################")
# print("Confusion Matrix")
# print(cm(Test_Y, predictions))
# print('\n')
# print("Classification Report ")
# print(classification_report(Test_Y, predictions.astype(np.int)))
# print("Random forest Accuracy Score ->",
#       metrics.accuracy_score(Test_Y, predictions.astype(np.int)))


# fit the training dataset on the NB classifier
# Naive = naive_bayes.MultinomialNB()
# Naive.fit(Train_X_Tfidf, Train_Y)
# predict the labels on validation dataset
# predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
# print("Naive Bayes Accuracy Score -> ",
#       accuracy_score(predictions_NB, Test_Y)*100)
# print("Confusion Matrix")
# print(cm(Test_Y, predictions_NB))
# print('\n')
# print("Classification Report ")
# print(classification_report(Test_Y, predictions_NB))
# print("Naive Bayes Accuracy Score ->",
#       metrics.accuracy_score(Test_Y, predictions_NB))
# plot_confusion_matrix(cm(Test_Y, predictions_NB), classes=['Rape', 'Theft','Assault','Murder','Statutory'], title='Confusion matrix')
# plt.show()


# logreg = LogisticRegression(n_jobs=1, C=1e5)
# logreg.fit(Train_X_Tfidf, Train_Y)
# y_pred = logreg.predict(Test_X_Tfidf)
# y_pred = logreg.predict(Test_X1_Tfidf)
# print("_____________++++++++++++++++####################")

# print(y_pred)

# live_data_classes = y_pred

# print("_____________++++++++++++++++####################")

# print("Confusion Matrix")
# print(cm(Test_Y, y_pred))
# print('\n')
# print("Classification Report ")
# print(classification_report(Test_Y, y_pred))
# print("log reg  Accuracy Score ->", metrics.accuracy_score(Test_Y, y_pred))

# plot_confusion_matrix(cm(Test_Y, y_pred), classes=['Rape', 'Theft','Assault','Murder','Statutory'], title='Confusion matrix')
# plt.show()

# model = LinearSVC()
# model.fit(Train_X_Tfidf, Train_Y)
# print('----')
# y_pred_svc = model.predict(Test_X_Tfidf)
# print("Confusion Matrix")
# print(cm(Test_Y, y_pred_svc))
# print('\n')
# print("Classification Report ")
# print(classification_report(Test_Y, y_pred_svc))
# print("linear SVC Accuracy Score  Count vect->",
#       metrics.accuracy_score(Test_Y, y_pred_svc))

# plot_confusion_matrix(cm(Test_Y, y_pred_svc), classes=['Rape', 'Theft','Assault','Murder','Statutory'], title='Confusion matrix')
# plt.show()


vec = CountVectorizer()
data = vec.fit_transform(crime_data['original_text'])
vector = vec.transform(crime_data['original_text'])
# summarize encoded vector
print(vector.shape)

# Train_X_count_vec = vec.transform(Train_X.values.astype('U'))
# Test_X_count_vec = vec.transform(Test_X.values.astype('U'))


# gb_clf = GradientBoostingClassifier(
#     n_estimators=20, learning_rate=0.5, max_features=100, max_depth=55, random_state=0)
# gb_clf.fit(Train_X_count_vec, Train_Y)


# predictions = gb_clf.predict(Test_X_count_vec)
# print(cm(Test_Y, predictions))
# print('\n')
# print("Classification Report ")
# print(classification_report(Test_Y, predictions))
# print("gradiant forest Accuracy Score ->",
#       metrics.accuracy_score(Test_Y, predictions))


# Naive = naive_bayes.MultinomialNB()
# Naive.fit(Train_X_count_vec, Train_Y)
# predict the labels on validation dataset
# predictions_NB = Naive.predict(Test_X_count_vec)
# Use accuracy_score function to get the accuracy
# print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
# print("Confusion Matrix")
# print(cm(Test_Y, predictions_NB))
# print('\n')
# print("Classification Report ")
# print(classification_report(Test_Y, predictions_NB))
# print("Naive Bayes Accuracy Score using count vect ->",
#       metrics.accuracy_score(Test_Y, predictions_NB))

# plot_confusion_matrix(cm(Test_Y, predictions_NB), classes=['Rape', 'Theft','Assault','Murder','Statutory'], title='Confusion matrix')
# plt.show()


# logreg = LogisticRegression(n_jobs=1, C=1e5)
# logreg.fit(Train_X_count_vec, Train_Y)
# y_pred = logreg.predict(Test_X_count_vec)
# print("Confusion Matrix")
# print(cm(Test_Y, y_pred))
# print('\n')
# print("Classification Report ")
# print(classification_report(Test_Y, y_pred))
# print("log reg  Accuracy Score Count Vect ->",
#       metrics.accuracy_score(Test_Y, y_pred))

# plot_confusion_matrix(cm(Test_Y, y_pred), classes=['Rape', 'Theft','Assault','Murder','Statutory'], title='Confusion matrix')
# plt.show()

# model = LinearSVC()
# model.fit(Train_X_count_vec, Train_Y)
# print('----')
# y_pred_svc = model.predict(Test_X_count_vec)
# print("Confusion Matrix")
# print(cm(Test_Y, y_pred_svc))
# print('\n')
# print("Classification Report ")
# print(classification_report(Test_Y, y_pred_svc))
# print("linear SVC Accuracy Score ->",
#       metrics.accuracy_score(Test_Y, y_pred_svc))

# plot_confusion_matrix(cm(Test_Y, y_pred_svc), classes=['Rape', 'Theft','Assault','Murder','Statutory'], title='Confusion matrix')
# plt.show()


# vec_transformed = CountVectorizer(min_df=200, max_df=150.0, lowercase=True)
# data = vec_transformed.fit_transform(crime_data['original_text'])


# Train_X_count_vec_tran = vec_transformed.transform(Train_X.values.astype('U'))
# Test_X_count_vec_tran = vec_transformed.transform(Test_X.values.astype('U'))


# # Naive_count = naive_bayes.MultinomialNB()
# # Naive_count.fit(Train_X_count_vec_tran, Train_Y)
# # predict the labels on validation dataset
# # predictions_NB_vec = Naive_count.predict(Test_X_count_vec_tran)
# # Use accuracy_score function to get the accuracy
# #print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
# # print("Confusion Matrix")
# # print(cm(Test_Y, predictions_NB_vec))
# # print('\n')
# # print("Classification Report ")
# # print(classification_report(Test_Y, predictions_NB_vec))
# # print("Naive Bayes Accuracy Score using count vect min df ->",
# #       metrics.accuracy_score(Test_Y, predictions_NB_vec))


# model = LinearSVC()
# model.fit(Train_X_count_vec_tran, Train_Y)
# y_pred_svc1 = model.predict(Test_X_count_vec_tran)
# print("Confusion Matrix")
# print(cm(Test_Y, y_pred_svc1))
# print('\n')
# print("Classification Report ")
# print(classification_report(Test_Y, y_pred_svc1))
# print("linear SVC Accuracy Score  Count vect->",
#       metrics.accuracy_score(Test_Y, y_pred_svc1))

# plot_confusion_matrix(cm(Test_Y, y_pred_svc1), classes=['Rape', 'Theft','Assault','Murder','Statutory'], title='Confusion matrix')
# plt.show()


# # logreg1 = LogisticRegression(n_jobs=1, C=1e5)
# # logreg1.fit(Train_X_count_vec_tran, Train_Y)
# # y_pred_vect_1 = logreg1.predict(Test_X_count_vec_tran)
# # print(cm(Test_Y, y_pred_vect_1))
# # print('\n')
# # print("Classification Report ")
# # print(classification_report(Test_Y, y_pred_vect_1))
# # print("log reg  Accuracy Score ->", metrics.accuracy_score(Test_Y, y_pred_vect_1))

# # kmodel = KNeighborsClassifier(n_neighbors=5)
# # kmodel.fit(Train_X_count_vec_tran, Train_Y)
# # y_pred_vect_1 = kmodel.predict(Test_X_count_vec_tran)
# # print(cm(Test_Y, y_pred_vect_1))
# # print('\n')
# # print("Classification Report ")
# # print(classification_report(Test_Y, y_pred_vect_1))
# # print("knn  Accuracy Score ->", metrics.accuracy_score(Test_Y, y_pred_vect_1))

# # balanced_x, balanced_y = balance_classes(
# #     crime_data['original_text'], crime_data['class_value'])
# # print(Counter(balanced_y))

# # Train_X_balanced, Test_X_balanced, Train_Y_balanced, Test_Y_balanced = model_selection.train_test_split(
# #     balanced_x, balanced_y, test_size=0.3)


# # vec_transformed = CountVectorizer(min_df=200, max_df=150.0, lowercase=True)
# # data = vec_transformed.fit_transform(balanced_x)


# # Train_X_count_vec_bal = vec_transformed.transform(Train_X_balanced)
# # Test_X_count_vec_bal = vec_transformed.transform(Test_X_balanced)


# # Naive_count = naive_bayes.MultinomialNB()
# # Naive_count.fit(Train_X_count_vec_bal,Train_Y_balanced)
# # # predict the labels on validation dataset
# # predictions_NB_vec = Naive_count.predict(Test_X_count_vec_bal)
# # # Use accuracy_score function to get the accuracy
# # #print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
# # print(cm(Test_Y_balanced, predictions_NB_vec))
# # print('\n')
# # print("Classification Report ")
# # print(classification_report(Test_Y_balanced, predictions_NB_vec))
# # print("Naive Bayes Accuracy Score using count vect min df ->",metrics.accuracy_score(Test_Y_balanced, predictions_NB_vec))
