import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gensim as gs
%pylab inline

#part 1 - data preparation 
text=[]
clas = []
classname = ["CEO", "Chairman", "Governance", "Highlights", "Remuniration"]
for item in classname:
    for file in os.listdir(r"/media/mi_air/Transcend/Folder Shared to Participants/S4-text-classification (2)/SummerSchoolSharedTask/Training/TrainingData/" +item ):
            filename = r"/media/mi_air/Transcend/Folder Shared to Participants/S4-text-classification (2)/SummerSchoolSharedTask/Training/TrainingData/" +item + "//"+file
            fl = open(filename, "r", encoding="utf8").read()
            fl = re.sub("\n", " ", fl)
            text.append(fl)
            clas.append(item)

data = pd.DataFrame(clas, columns=['class'])
data["text"] = text

from sklearn.utils import shuffle
data = shuffle(data)
data["text"].fillna("0")

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import *
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

y = data["class"]
X = data["text"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

rs = 42

#logit regression 
clf = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,3), analyzer='word', max_features=200)),
    ('tfidf', TfidfTransformer(sublinear_tf=True)),
    #('reducer', TruncatedSVD(n_components=Val3)),
    ('clf',  LogisticRegression(random_state=rs)),
    ])
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(classif)
print("Precision: {0:6.2f}".format(precision_score(y_test, predictions, average='macro')))
print("Recall: {0:6.2f}".format(recall_score(y_test, predictions, average='macro')))
print("F1-measure: {0:6.2f}".format(f1_score(y_test, predictions, average='macro')))
print("Accuracy: {0:6.2f}".format(accuracy_score(y_test, predictions)))
print(classification_report(y_test, predictions))
labels = clf.classes_
sns.heatmap(data=confusion_matrix(y_test, predictions), annot=True, fmt="d", cbar=False, xticklabels=labels, yticklabels=labels)
plt.title("Confusion matrix")
plt.show()


#logit regression with best params by grid search
clf2 = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,3), analyzer='word', max_features=200)),
    ('tfidf', TfidfTransformer(sublinear_tf=True)),
    #('reducer', TruncatedSVD(n_components=Val3)),
    ('clf',  LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=42, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)),
    ])
clf2.fit(X_train, y_train)
predictions = clf2.predict(X_test)
print(classif)
print("Precision: {0:6.2f}".format(precision_score(y_test, predictions, average='macro')))
print("Recall: {0:6.2f}".format(recall_score(y_test, predictions, average='macro')))
print("F1-measure: {0:6.2f}".format(f1_score(y_test, predictions, average='macro')))
print("Accuracy: {0:6.2f}".format(accuracy_score(y_test, predictions)))
print(classification_report(y_test, predictions))
labels = clf.classes_
sns.heatmap(data=confusion_matrix(y_test, predictions), annot=True, fmt="d", cbar=False, xticklabels=labels, yticklabels=labels)
plt.title("Confusion matrix")
plt.show()

#loading test data
files=[]
texts=[]

for file in os.listdir(r"/media/mi_air/Transcend/Folder Shared to Participants/S4-text-classification (2)/SummerSchoolSharedTask/Testing/TestingData"):
    filename = r"/media/mi_air/Transcend/Folder Shared to Participants/S4-text-classification (2)/SummerSchoolSharedTask/Testing/TestingData/" + file
    test = open(filename, "r", encoding="utf8").read()
    test = re.sub("\n", " ", test)
    files.append(file)
    texts.append(test)
testdata = pd.DataFrame(files, columns=['filename'])
testdata["text"] = texts

#predicting classes for test data
predictions2 = clf.predict(testdata["text"])
outfile = open(r"/home/mi_air/testing_logitregression.txt", "w", encoding="utf8")
for i in range(len(testdata)):
    stroka = testdata["filename"].iloc[i] + "\t" + predictions2[i]
    outfile.write(stroka + "\n")

outfile2 = open(r"/home/mi_air/testing_logitregression_grids.txt", "w", encoding="utf8")
predictions3 = clf2.predict(testdata["text"])
for i in range(len(testdata)):
    stroka = testdata["filename"].iloc[i] + "\t" + predictions3[i]
    outfile2.write(stroka + "\n")

