"""
This python file is used a helper to
    - Get the basis testing results for our approaches.
    - Get the 20-fold cross validation results for our novel approach.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pickle

mergedDatasetFiltered = pd.read_csv(str(os.getcwd()) + '/My Model/Paper 3 All Features Filtered.csv')  # Get the csv file that contains all features

label_set = mergedDatasetFiltered["label_x"]  # Get label set

mergedDatasetFiltered = mergedDatasetFiltered.drop(["label_x", "label_y", "total_tweets"], axis=1)  # Drop all labels

X_train, X_test, y_train, y_test = train_test_split(mergedDatasetFiltered, label_set, test_size=0.2)  # Split the data into training and testing data

"""
For every classifier below, the same operations are done for each of them.
These operations are listed as:
    - Print the name of the classifier
    - Get the classifier into a variable
    - Train the classifier
    - Get the predictions of the classifier on the testing set
    - Print the accuracy score
    - Print the confusion matrix
    - Print the AUC score
    - As an addition, pickle.dump(rf, open('MODEL NAME', 'wb')) can be used to extract the machine learning model
"""

print("Random Forest")  # Print the name of the classifier
rf = RandomForestClassifier(n_estimators=150)  # Get the classifier into a variable
rf.fit(X_train, y_train)  # Train the classifier
prediction = rf.predict(X_test)  # Get the predictions of the classifier on the testing set
print(accuracy_score(y_test, prediction))  # Print the accuracy score
print(confusion_matrix(y_test, prediction))  # Print the confusion matrix
print(roc_auc_score(y_test, prediction))  # Print the AUC score

# pickle.dump(rf, open('myModel', 'wb'))  # Extract model

print("Adaboost")  # Print the name of the classifier
clf = AdaBoostClassifier(n_estimators=100, random_state=0)  # Get the classifier into a variable
clf.fit(X_train, y_train)  # Train the classifier
prediction = clf.predict(X_test)  # Get the predictions of the classifier on the testing set
print(accuracy_score(y_test, prediction))  # Print the accuracy score
print(confusion_matrix(y_test, prediction))  # Print the confusion matrix
print(roc_auc_score(y_test, prediction))  # Print the AUC score

# pickle.dump(rf, open('myModelADA', 'wb'))  # Extract model

print("Xgboost")  # Print the name of the classifier
xgb = XGBClassifier()  # Get the classifier into a variable
xgb.fit(X_train, y_train)  # Train the classifier
prediction = xgb.predict(X_test)  # Get the predictions of the classifier on the testing set
print(accuracy_score(y_test, prediction))  # Print the accuracy score
print(confusion_matrix(y_test, prediction))  # Print the confusion matrix
print(roc_auc_score(y_test, prediction))  # Print the AUC score

# pickle.dump(rf, open('myModelXGB', 'wb'))  # Extract model

print("Logistic Regression")  # Print the name of the classifier
Linreg = LogisticRegression().fit(X_train, y_train)  # Get the classifier into a variable, train the classifier
prediction = Linreg.predict(X_test)  # Get the predictions of the classifier on the testing set
print(accuracy_score(y_test, prediction))  # Print the accuracy score
print(confusion_matrix(y_test, prediction))  # Print the confusion matrix
print(roc_auc_score(y_test, prediction))  # Print the AUC score


print("Naive-bayes")  # Print the name of the classifier
bayes = GaussianNB()  # Get the classifier into a variable
bayes.fit(X_train,y_train)  # Train the classifier
prediction = bayes.predict(X_test)  # Get the predictions of the classifier on the testing set
print(accuracy_score(y_test, prediction))  # Print the accuracy score
print(confusion_matrix(y_test, prediction))  # Print the confusion matrix
print(roc_auc_score(y_test, prediction))  # Print the AUC score


print("k-NN")  # Print the name of the classifier
neigh = KNeighborsClassifier(n_neighbors=6)  # Get the classifier into a variable
neigh.fit(X_train,y_train)  # Train the classifier
prediction = neigh.predict(X_test)  # Get the predictions of the classifier on the testing set
print(accuracy_score(y_test, prediction))  # Print the accuracy score
print(confusion_matrix(y_test, prediction))  # Print the confusion matrix
print(roc_auc_score(y_test, prediction))  # Print the AUC score

# IN ORDER TO DO 20-FOLD CROSS VALIDATION, COMMENT OUT ONE OF THE BELOW PIECE OF CODE AND USE THE VARIABLE FOR THE CHOSEN CLASSIFIER

# THIS CODE GIVES THE 20-FOLD CROSS VALIDATION ACCURACY RESULT OF THE CLASSIFIER
"""scores = cross_val_score(rf, mergedDatasetFiltered, label_set, cv=20)  # Get all 20-fold cross validation results
print(scores)  # Print all the accuracy scores
print(mean(scores))   # Print the mean accuracy score"""

# THIS CODE GIVES THE 20-FOLD CROSS VALIDATION CONFUSION MATRIX RESULTS OF THE CLASSIFIER

"""y_pred = cross_val_predict(rf, mergedDatasetFiltered, label_set, cv=20)
conf_mat = confusion_matrix(label_set, y_pred)
print(conf_mat)"""
