# -*- coding: utf-8 -*-

"""
regression the data using ols
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import parameters
import load_data
import preprocess
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas
import single_feature_distribution
import itertools
from sklearn.metrics import auc


def display_matrix(confusion_matrix):
    confusion_matrix = np.array(confusion_matrix)
    classes = [0, 1]
    plt.figure()
    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = np.max(confusion_matrix) / 2
    for i, j in itertools.product(range(len(confusion_matrix)),
                                  range(len(confusion_matrix[0]))):
        plt.text(j, i, str(confusion_matrix[i][j]),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i][j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def get_best_divide_line(Y_pred, Y_test, count, size, show_image=False):
    docu_ROC = []
    X = [i for i in range(size + 1)]
    confusion_matrix = [[0, 0], [0, 0]]
    data_plot = {
        "predict": [i for i in range(size)],
        "true": [i for i in range(size)]
    }
    data = pandas.DataFrame(data_plot)
    for i in range(size):
        data.at[i, "predict"] = Y_pred[i]
        data.at[i, "true"] = Y_test[Y_test.columns[0]].iat[i]
    data.sort_values("predict", inplace=True, ascending=False)
    TPR_FPR = pandas.DataFrame(index=range(size + 1), columns=("TP", "FP"))
    F, T = size - count, count
    FP, TP = 0, 0
    TPR_FPR.iloc[0] = [FP, TP]
    docu_ROC.append(0)
    for index in range(size):
        if show_image:
            standard = single_feature_distribution.get_n_largest(Y_pred, index + 1)
            for i in range(size):
                if Y_pred[i] >= standard and Y_test[Y_test.columns[0]].iat[i] == 1:
                    confusion_matrix[1][1] += 1
                elif Y_pred[i] < standard and Y_test[Y_test.columns[0]].iat[i] == 0:
                    confusion_matrix[0][0] += 1
                elif Y_pred[i] < standard and Y_test[Y_test.columns[0]].iat[i] == 1:
                    confusion_matrix[1][0] += 1
                elif Y_pred[i] >= standard and Y_test[Y_test.columns[0]].iat[i] == 0:
                    confusion_matrix[0][1] += 1
                else:
                    continue
            display_matrix(confusion_matrix)
        TPR_FPR.iloc[index + 1] = [TP, FP]
        docu_ROC.append(TP * (1 - FP))
        if Y_test[Y_test.columns[0]].iat[index] == 1:
            TP += 1 / T
        else:
            FP += 1 / F
    AUC = auc(TPR_FPR["FP"], TPR_FPR["TP"])
    plt.figure()
    # plt.scatter(x=TPR_FPR["FP"], y=TPR_FPR["TP"], label="(FPR,TPR)", color="blueviolet")
    plt.plot(TPR_FPR["FP"], TPR_FPR["TP"], "blueviolet", label="AUC = %0.4f" % AUC)  # blueviolet
    plt.legend(loc="lower right")
    plt.title("Receiver Operating Characteristic")
    plt.plot([(0, 0), (1, 1)], "r--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 01.01])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()
    print("AUC =", AUC)
    plt.figure()
    plt.plot(X, docu_ROC, "orangered")
    plt.xlabel("X from 0 to" + str(size + 1))
    plt.ylabel("TP * (1 - FP)")
    plt.title("find the best standard")
    plt.show()
    best_index = X[np.argmax(docu_ROC)]
    return single_feature_distribution.get_n_largest(Y_pred, best_index)


def plot_pred(data, model, standard, name):
    data = data.sample(frac=parameters.SAMPLE_RATIO).reset_index(drop=True)
    print("data.shape ->", data.shape)
    select_col = []
    col_size = len(data.columns)
    for i in range(col_size - 1):
        select_col.append(data.columns[i])
    test = pandas.DataFrame(data, columns=select_col)
    print("test.shape ->", test.shape)
    pred = model.predict(np.array(test))
    data_plot = {
        "1": [i for i in range(data.shape[0])],
        "2": [i for i in range(data.shape[0])],
        "3": [i for i in range(data.shape[0])],
        "4": [i for i in range(data.shape[0])],
        "real_data": [i for i in range(data.shape[0])],
        "predict": [i for i in range(data.shape[0])]
    }
    data_plot = pandas.DataFrame(data_plot)
    for i in range(data.shape[0]):
        # just for better plotting, no sense
        data_plot.at[i, "1"] = data.at[i, data.columns[1]]
        data_plot.at[i, "2"] = data.at[i, data.columns[2]]
        data_plot.at[i, "3"] = data.at[i, data.columns[3]]
        data_plot.at[i, "4"] = data.at[i, data.columns[4]]
        data_plot.at[i, "real_data"] = data[data.columns[data.shape[1] - 1]].iat[i]
        if pred[i] >= standard:
            data_plot.at[i, "predict"] = 1
        elif pred[i] < standard:
            data_plot.at[i, "predict"] = 0
        else:
            continue
    data_plot_pred = pandas.DataFrame(data_plot, columns=["1", "2", "3", "4", "predict"])
    data_plot_real = pandas.DataFrame(data_plot, columns=["1", "2", "3", "4", "real_data"])

    plt.figure()
    plt.subplot(1, 2, 1)
    pandas.plotting.andrews_curves(data_plot_pred, data_plot_pred.columns[data_plot_pred.shape[1] - 1],
                                   color=["moccasin", "cyan"], alpha=0.80)
    plt.title(name + " andrews_predict")
    plt.subplot(1, 2, 2)
    pandas.plotting.andrews_curves(data_plot_real, data_plot_real.columns[data_plot_real.shape[1] - 1],
                                   color=["palegreen", "fuchsia"], alpha=0.80)
    plt.title(name + " andrews_real_data")
    plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    pandas.plotting.parallel_coordinates(data_plot_pred, data_plot_pred.columns[data_plot_pred.shape[1] - 1],
                                         color=["palegreen", "fuchsia"], alpha=0.80)
    plt.title(name + " parallel_real_data")
    plt.subplot(1, 2, 2)
    pandas.plotting.parallel_coordinates(data_plot_real, data_plot_real.columns[data_plot_real.shape[1] - 1],
                                         color=["skyblue", "red"], alpha=0.80)
    plt.title(name + " parallel_predict")
    plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    pandas.plotting.radviz(data_plot_pred, data_plot_pred.columns[data_plot_pred.shape[1] - 1],
                           color=["skyblue", "red"], alpha=0.80)
    plt.title(name + " radviz predict")
    plt.subplot(1, 2, 2)
    pandas.plotting.radviz(data_plot_real, data_plot_real.columns[data_plot_real.shape[1] - 1],
                           color=["moccasin", "cyan"], alpha=0.80)
    plt.title(name + " radviz real_data")
    plt.show()
    print("linear regression plot done.")


def ols_analysis(data, feature, target, mode):
    print("feature ->", feature.columns)
    print("target ->", target.columns)
    X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size=0.35, random_state=1)
    Linear = LinearRegression(fit_intercept=True, n_jobs=None, positive=False)
    X_train, Y_train = preprocess.un_balance(X_train, Y_train)
    Linear.fit(X_train, Y_train)
    print("Slope ->")
    print(Linear.coef_)
    print("Intercept ->")
    print(Linear.intercept_)
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
    if mode:
        standard = get_best_divide_line(Y_pred, Y_test, count, size, show_image=False)
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
    plot_pred(data, Linear, standard, "ols")


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path)
    ols_analysis(end_off, end_off_feature, end_off_target, mode=True)
