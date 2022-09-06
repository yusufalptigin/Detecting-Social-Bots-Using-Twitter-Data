import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('paper2_features.csv')

label_set = dataset["label"]

dataset = dataset.drop(["label"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(dataset, label_set, test_size=0.2)

print("Random Forest")
rf = RandomForestClassifier(n_estimators=150)
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))
prediction = rf.predict(X_test)
print(accuracy_score(y_test, prediction))
print(confusion_matrix(y_test, prediction))

print("Adaboost")
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
prediction = clf.predict(X_test)
print(accuracy_score(y_test, prediction))
print(confusion_matrix(y_test, prediction))


print("Xgboost")
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
print(xgb.score(X_test, y_test))
prediction = xgb.predict(X_test)
print(accuracy_score(y_test, prediction))
print(confusion_matrix(y_test, prediction))


print("logistic regression")
Linreg = LogisticRegression().fit(X_train, y_train)
print(Linreg.score(X_test, y_test))
prediction = Linreg.predict(X_test)
print(accuracy_score(y_test, prediction))
print(confusion_matrix(y_test, prediction))


print("SVM")
supvecmac = svm.SVC().fit(X_train, y_train)
print(supvecmac.score(X_test, y_test))
prediction = supvecmac.predict(X_test)
print(accuracy_score(y_test, prediction))
print(confusion_matrix(y_test, prediction))


print("Naive-bayes")
bayes = GaussianNB()
bayes.fit(X_train,y_train)
print(bayes.score(X_test, y_test))
prediction = bayes.predict(X_test)
print(accuracy_score(y_test, prediction))
print(confusion_matrix(y_test, prediction))


print("KNN")
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
print(neigh.score(X_test, y_test))
prediction = neigh.predict(X_test)
print(accuracy_score(y_test, prediction))
print(confusion_matrix(y_test, prediction))