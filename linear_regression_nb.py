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
from sklearn.cross_decomposition import PLSRegression
import liner_regression_ols


def pls_regression(data, feature, target):
    print("feature ->", feature.columns)
    print("target ->", target.columns)
    X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size=0.35, random_state=1)
    Linear = GaussianNB()
    Linear.fit(np.array(X_train), np.array(Y_train))
    Y_pred = Linear.predict(X_test)
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
    standard = liner_regression_ols.get_best_divide_line(Y_pred, Y_test, count, size, show_image=False)
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
    liner_regression_ols.plot_pred(data, Linear, standard, "pls")


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path)
    pls_regression(end_off, end_off_feature, end_off_target)