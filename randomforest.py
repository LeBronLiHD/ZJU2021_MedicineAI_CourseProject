# -*- coding: utf-8 -*-

"""
use random forest to train model
"""

import liner_regression_ols
import parameters
import load_data
import preprocess
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
import single_feature_distribution
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


def random_forest(data, feature, target, mode=True):
    print("feature ->", feature.columns)
    print("target ->", target.columns)
    X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size=0.35, random_state=1)
    forest = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=10, max_features=None, n_jobs=6, verbose=1)
    X_train, Y_train= preprocess.un_balance(X_train, Y_train, ratio="minority")
    print("X_train.shape ->", X_train.shape)
    print("Y_train.shape ->", Y_train.shape)
    print("len(Y_train) ->", np.shape(np.array(Y_train.values.ravel())))
    forest.fit(np.array(X_train), np.array(Y_train.values.ravel()))
    # define evaluation procedure
    print("model done.")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    # evaluate model
    scores = cross_val_score(forest, np.array(X_test), np.array(Y_test.values.ravel()), scoring="roc_auc", cv=cv,
                             n_jobs=6)
    # summarize performance
    print("mean roc_auc: %.8f" % np.mean(scores))
    # Predict the response for test dataset
    Y_pred = forest.predict(X_test)
    print("explained_variance_score ->",
          metrics.explained_variance_score(Y_test, Y_pred))
    print("mean_absolute_error ->",
          metrics.mean_absolute_error(Y_test, Y_pred))
    print("mean_squared_error ->",
          metrics.mean_squared_error(Y_test, Y_pred))
    print("mean_standard_error ->",
          np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
    size = len(Y_test)
    count = 0
    for i in range(size):
        if Y_test[Y_test.columns[0]].iat[i] == 1:
            count += 1
    print(count, size - count)
    if mode:
        standard = liner_regression_ols.get_best_divide_line(Y_pred, Y_test, count, size, show_image=False)
    else:
        standard = single_feature_distribution.get_n_largest(Y_pred, count)
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
    liner_regression_ols.plot_pred(data, forest, standard, "forest")


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path,
                                                                                                       test_mode=True)
    random_forest(end_off, end_off_feature, end_off_target, mode=True)
