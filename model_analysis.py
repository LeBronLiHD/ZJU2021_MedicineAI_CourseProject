# -*- coding: utf-8 -*-

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
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score


def model_analysis(X_test, Y_test, model, data, mode, normal=True, pca=False, pca_model=None):
    if pca:
        X_test = pca_model.fit_transform(X_test.iloc[:, :-1])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    # evaluate model
    scores = cross_val_score(model, np.array(X_test), np.array(Y_test.values.ravel()),
                             scoring="roc_auc", cv=cv, n_jobs=6)
    # summarize performance
    print("mean roc_auc: %.8f" % np.mean(scores))
    Model_List_1_auc[mode] = np.mean(scores)
    # Predict the response for test dataset
    Y_pred = model.predict(X_test)
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
    Model_List_1_right_0[mode] = right_0_1[0] / (size - count)
    Model_List_1_right_1[mode] = right_0_1[1] / count
    Model_List_1_right_all[mode] = right / size
    if normal:
        data = preprocess.data_normalization(data, have_target=True)
    plot_pred(data, model, standard, Model_List_1[mode], pca=pca, pca_model=pca_model)


def plot_pred(data, model, standard, name, pca=False, pca_model=None):
    data = data.sample(frac=parameters.SAMPLE_RATIO).reset_index(drop=True)
    print("data.shape ->", data.shape)
    select_col = []
    col_size = len(data.columns)
    for i in range(col_size - 1):
        select_col.append(data.columns[i])
    test = pandas.DataFrame(data, columns=select_col)
    print("test.shape ->", test.shape)
    if pca:
        test = pca_model.fit_transform(test.iloc[:, :-1])
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


Model_List_1 = [
    "decision_tree",
    "xgboost",
    "random_forest",
    "pca-svm",
    "neural_network",
    "convolution_nn"
]

Model_List_1_time = [
    0,
    0,
    0,
    0,
    0,
    0
]

Model_List_1_auc = [
    0,
    0,
    0,
    0,
    0,
    0
]

Model_List_1_right_1 = [
    0,
    0,
    0,
    0,
    0,
    0
]

Model_List_1_right_0 = [
    0,
    0,
    0,
    0,
    0,
    0
]

Model_List_1_right_all = [
    0,
    0,
    0,
    0,
    0,
    0
]
