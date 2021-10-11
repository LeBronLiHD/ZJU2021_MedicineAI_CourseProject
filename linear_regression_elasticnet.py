# -*-  coding: utf-8 -*-

"""
regression the data using ElasticNet
"""


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import parameters
import load_data
import preprocess
from matplotlib import pyplot as plt
import pandas
import single_feature_distribution


def plot_pred(data, model, standard):
    data = data.sample(frac=parameters.SAMPLE_RATIO).reset_index(drop=True)
    print("data.shape ->", data.shape)
    select_col = []
    col_size = len(data.columns)
    for i in range(col_size - 1):
        select_col.append(data.columns[i])
    test = pandas.DataFrame(data, columns=select_col)
    print("test.shape ->", test.shape)
    pred = model.predict(test)
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
                                   color=["orange", "cyan"], alpha=0.7)
    plt.title("andrews_predict")
    plt.subplot(1, 2, 2)
    pandas.plotting.andrews_curves(data_plot_real, data_plot_real.columns[data_plot_real.shape[1] - 1],
                                   color=["chartreuse", "fuchsia"], alpha=0.7)
    plt.title("andrews_real_data")
    plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    pandas.plotting.parallel_coordinates(data_plot_pred, data_plot_pred.columns[data_plot_pred.shape[1] - 1],
                                         color=["chartreuse", "fuchsia"], alpha=0.7)
    plt.title("parallel_real_data")
    plt.subplot(1, 2, 2)
    pandas.plotting.parallel_coordinates(data_plot_real, data_plot_real.columns[data_plot_real.shape[1] - 1],
                                         color=["deepskyblue", "orangered"], alpha=0.7)
    plt.title("parallel_predict")
    plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    pandas.plotting.radviz(data_plot_pred, data_plot_pred.columns[data_plot_pred.shape[1] - 1],
                           color=["deepskyblue", "orangered"], alpha=0.7)
    plt.title("radviz predict")
    plt.subplot(1, 2, 2)
    pandas.plotting.radviz(data_plot_real, data_plot_real.columns[data_plot_real.shape[1] - 1],
                           color=["orange", "cyan"], alpha=0.7)
    plt.title("radviz real_data")
    plt.show()
    print("ols plot done.")


def elastic_net(data, feature, target):
    print("feature ->", feature.columns)
    print("target ->", target.columns)
    X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size=0.2, random_state=1)
    Linear = LinearRegression()
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
        if Y_test[Y_test.columns[0]].iat[i] == 1 == 1:
            count += 1
    print(count, size - count)
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
    print("right ratio =", right / size)
    print("right_0_1 ->", right_0_1)
    print("error_0_1 ->", error_0_1)
    plot_pred(data, Linear, standard)


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path)
    elastic_net(end_off, end_off_feature, end_off_target)
