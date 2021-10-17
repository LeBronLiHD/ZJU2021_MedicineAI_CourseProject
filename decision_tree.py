# -*- coding: utf-8 -*-

"""
use decision tree to train model
Decision Tree is a white box type of ML algorithm. It shares internal decision-making logic,
which is not available in the black box type of algorithms such as Neural Network

1. Select the best attribute using Attribute Selection Measures(ASM) to split the records.
2. Make that attribute a decision node and breaks the dataset into smaller subsets.
3. Starts tree building by repeating this process recursively for each child until one of the condition will match:
    a. All the tuples belong to the same attribute value.
    b. There are no more remaining attributes.
    c. There are no more instances.
"""

import liner_regression_ols
import parameters
import load_data
import preprocess
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas
import single_feature_distribution
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score


def decision_tree(data, feature, target):
    print("feature ->", feature.columns)
    print("target ->", target.columns)
    feature = preprocess.data_normalization(feature)
    X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size=0.35, random_state=1)
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="gini", splitter="best", max_features=None)
    # Train Decision Tree Classifer
    X_train, Y_train= preprocess.un_balance(X_train, Y_train, ratio="minority", mode=1, ensemble=False)
    clf.fit(X_train, Y_train)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=0)
    # evaluate model
    scores = cross_val_score(clf, np.array(X_test), np.array(Y_test.values.ravel()), scoring="roc_auc", cv=cv, n_jobs=4)
    # summarize performance
    print("mean roc_auc: %.8f" % np.mean(scores))
    # Predict the response for test dataset
    Y_pred = clf.predict(X_test)
    print("explained_variance_score ->",
          metrics.explained_variance_score(Y_test, Y_pred))
    print("mean_absolute_error ->",
          metrics.mean_absolute_error(Y_test, Y_pred))
    print("mean_squared_error ->",
          metrics.mean_squared_error(Y_test, Y_pred))
    print("mean_standard_error ->",
          np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
    print("accuracy:", metrics.accuracy_score(Y_test, Y_pred))
    size = len(Y_test)
    count = 0
    for i in range(size):
        if Y_test[Y_test.columns[0]].iat[i] == 1:
            count += 1
    print(count, size - count)
    standard = 0.5
    for i in range(size):
        if Y_pred[i] != 0 and Y_pred[i] != 1:
            print("fuck", end=" ")
    print("standard =", standard)
    right = 0
    right_0_1 = [0, 0]
    error_0_1 = [0, 0]
    for i in range(size):
        if Y_pred[i] >= standard and Y_test[Y_test.columns[0]].iat[i] == 1:
            right_0_1[1] += 1
            right += 1
        elif Y_pred[i] < standard and Y_test[Y_test.columns[0]].iat[i] == 0:
            right_0_1[0] += 1
            right += 1
        elif Y_pred[i] < standard and Y_test[Y_test.columns[0]].iat[i] == 1:
            error_0_1[1] += 1
        elif Y_pred[i] >= standard and Y_test[Y_test.columns[0]].iat[i] == 0:
            error_0_1[0] += 1
        else:
            continue
    print("right =", right)
    print("fault =", size - right)
    print("overall right ratio =", right / size)
    print("0 right ratio =", right_0_1[0] / (size - count))
    print("1 right ratio =", right_0_1[1] / count)
    print("right_0_1 ->", right_0_1)
    print("error_0_1 ->", error_0_1)
    data = preprocess.data_normalization(data, have_target=True)
    liner_regression_ols.plot_pred(data, clf, standard, "de-tree")


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path,
                                                                                                       test_mode=True)
    decision_tree(end_off, end_off_feature, end_off_target)
