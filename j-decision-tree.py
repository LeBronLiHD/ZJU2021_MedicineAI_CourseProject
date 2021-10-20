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
import model_analysis
from sklearn.decomposition import PCA
import time


def decision_tree(end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target):
    end_off, end_off_feature, merge, merge_feature = \
        preprocess.data_normalization(end_off, have_target=True), \
        preprocess.data_normalization(end_off_feature, have_target=False), \
        preprocess.data_normalization(merge, have_target=True), \
        preprocess.data_normalization(merge_feature, have_target=False)
    model, pca_model = train_decision_tree(merge_feature, merge_target, pca=True, n_com=30)
    model_analysis.model_analysis(end_off_feature, end_off_target, model, end_off, 0, normal=True, pca=True,
                                  pca_model=pca_model)


def train_decision_tree(X_train, Y_train, pca=False, n_com=30):
    print("decision tree training...")
    init_time = time.time()
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="scale", splitter="best", max_features=None)
    # Train Decision Tree Classifer
    X_train, Y_train= preprocess.un_balance(X_train, Y_train, ratio="minority", mode=1, ensemble=False)
    PCA_model = None
    if pca:
        PCA_model = PCA(n_components=n_com)
        # X = model.fit_transform(data_selected.iloc[:, :-1])
        X_train = PCA_model.fit_transform(X_train.iloc[:, :-1])
    clf.fit(X_train, Y_train)
    print("decision tree done, time ->", time.time() - init_time)
    model_analysis.Model_List_1_time[0] = time.time() - init_time
    return clf, PCA_model


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path,
                                                                                                       test_mode=False)
    decision_tree(end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target)
