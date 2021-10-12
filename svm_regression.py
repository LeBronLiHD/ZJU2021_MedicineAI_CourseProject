# -*- coding: utf-8 -*-

"""
data regression using GaussianNB
"""


from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import parameters
import load_data
import preprocess
from matplotlib import pyplot as plt
import pandas
import single_feature_distribution
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import liner_regression_ols


def cs_svm(data, feature, target, balance):
    print("feature ->", feature.columns)
    print("target ->", target.columns)
    feature = preprocess.data_normalization(feature)
    X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size=0.35, random_state=1)
    if balance:
        X_train, Y_train = preprocess.un_balance(X_train, Y_train, ratio=0.5)
    svc = SVC(gamma="scale", class_weight="balanced", tol=1e-10, degree=100)
    Linear = make_pipeline(StandardScaler(), svc)
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=0)
    # evaluate model
    scores = cross_val_score(svc, np.array(X_test), np.array(Y_test.values.ravel()), scoring="roc_auc", cv=cv, n_jobs=4)
    # summarize performance
    print("mean roc_auc: %.8f" % np.mean(scores))
    Linear.fit(np.array(X_train), np.array(Y_train))
    Y_pred = Linear.predict(np.array(X_test))
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
    standard = count/(size - count)
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
    data = preprocess.data_normalization(data)
    liner_regression_ols.plot_pred(data, Linear, standard, "cs-svm")


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path,
                                                                                                       test_mode=True)
    cs_svm(end_off, end_off_feature, end_off_target, balance=False)
