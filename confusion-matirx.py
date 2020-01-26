from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv('./data/confusion-matrix.csv')
train, test = train_test_split(df, test_size=0.3, random_state=0)

x_train = train.drop('class_value', axis=1)
y_train = train['class_value']

x_test = test.drop('class_value', axis=1)
y_test = test['class_value']


model = tree.DecisionTreeClassifier(random_state=1)
model.fit(x_train, y_train)


pred = model.predict(x_test)


cm = confusion_matrix(y_test, pred)
cm = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=[
                 'Predicted'], margins=True)

print(cm)
