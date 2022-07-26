import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = pd.read_csv('Movies_TV.txt', delimiter='\t')
# print(corpus.iloc[0])
X = corpus['Review']
# print(X)
Y = corpus['Label']

vec = TfidfVectorizer(max_df = 800, min_df = 5, ngram_range = (1,4), max_features = 200)
X = vec.fit_transform(X)
X = X.toarray()

# print(vec.vocabulary_)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.33, random_state=42)
# print(X_train[:1])

lc = SGDClassifier()
nbc = GaussianNB()
dtc = DecisionTreeClassifier()
knnc = KNeighborsClassifier()

lc.fit(X_train, y_train)
nbc.fit(X_train, y_train)
dtc.fit(X_train, y_train)
knnc.fit(X_train, y_train)

pred_y_lc = lc.predict(X_test)
pred_y_nbc = nbc.predict(X_test)
pred_y_dtc = dtc.predict(X_test)
pred_y_knnc = knnc.predict(X_test)

lc_acc = accuracy_score(y_test, pred_y_lc)
lc_prec = precision_score(y_test, pred_y_lc, average = 'macro')
lc_recall = recall_score(y_test, pred_y_lc, average = 'macro')
lc_f1 = f1_score(y_test, pred_y_lc, average='macro')

nbc_acc = accuracy_score(y_test, pred_y_nbc)
nbc_prec = precision_score(y_test, pred_y_nbc, average = 'macro')
nbc_recall = recall_score(y_test, pred_y_nbc, average = 'macro')
nbc_f1 = f1_score(y_test, pred_y_nbc, average='macro')

dtc_acc = accuracy_score(y_test, pred_y_dtc)
dtc_prec = precision_score(y_test, pred_y_dtc, average = 'macro')
dtc_recall = recall_score(y_test, pred_y_dtc, average = 'macro')
dtc_f1 = f1_score(y_test, pred_y_dtc, average='macro')

knnc_acc = accuracy_score(y_test, pred_y_knnc)
knnc_prec = precision_score(y_test, pred_y_knnc, average = 'macro')
knnc_recall = recall_score(y_test, pred_y_knnc, average = 'macro')
knnc_f1 = f1_score(y_test, pred_y_knnc, average='macro')

print("Linear Classifier: \nAccuracy Score: ", lc_acc, '\nPrecision Score:', lc_prec, '\nRecall Score:', lc_recall, '\nF1 Score:', lc_f1,'\n')
print("Naive Bayes Classifier: \nAccuracy Score: ", nbc_acc, '\nPrecision Score:', nbc_prec, '\nRecall Score:', nbc_recall, '\nF1 Score:', nbc_f1,'\n')
print("Decision Tree Classifier: \nAccuracy Score: ", dtc_acc, '\nPrecision Score:', dtc_prec, '\nRecall Score:', dtc_recall, '\nF1 Score:', dtc_f1,'\n')
print("KNN Classifier: \nAccuracy Score: ", knnc_acc, '\nPrecision Score:', knnc_prec, '\nRecall Score:', knnc_recall, '\nF1 Score:', knnc_f1,'\n')
