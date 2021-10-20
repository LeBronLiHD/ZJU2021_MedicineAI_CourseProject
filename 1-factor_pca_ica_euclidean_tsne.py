# -*- coding: utf-8 -*-

"""
Others algorithms to find the feature distribution
1. factor analysis
2. PCA
3. Fast ICA
4. euclidean distance
5. TSNE
"""

import seaborn as sns
from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.manifold import TSNE
import pandas
import load_data
import parameters
import preprocess
from sklearn.utils import shuffle
import single_feature_distribution
import matplotlib.pyplot as plt


def factor_analysis(data, important):
    data = data.sample(frac=parameters.SAMPLE_RATIO).reset_index(drop=True)
    print("data.shape ->", data.shape)
    print("important[0] ->", important[0])
    important_copy = []
    for i in range(len(important[0])):
        important_copy.append(important[0][i])
    important_copy.append(data.shape[1] - 1)
    select_col = []
    for i in range(len(important_copy)):
        select_col.append(data.columns[important_copy[i]])
    data_selected = pandas.DataFrame(data, columns=select_col)
    data_selected = preprocess.data_normalization(data_selected, have_target=True)
    print("data_selected.shape ->", data_selected.shape)
    print("data_selected.columns ->", data_selected.columns)
    model = decomposition.FactorAnalysis(n_components=parameters.N_COMPONENTS)
    X = model.fit_transform(data_selected.iloc[:, :-1].values)
    pos = pandas.DataFrame()
    pos['X'] = X[:, 0]
    pos['Y'] = X[:, 1]
    target = data.columns[data.shape[1] - 1]
    pos[target] = data_selected[target]
    axis = pos[pos[target] == 0].plot(kind='scatter', x='X', y='Y', color='green', label=0)
    pos[pos[target] == 1].plot(kind='scatter', x='X', y='Y', color='red', label=1, ax=axis)
    plt.title("factor_analysis")
    plt.show()


def PCA(data, important):
    data = data.sample(frac=parameters.SAMPLE_RATIO).reset_index(drop=True)
    print("data.shape ->", data.shape)
    print("important[0] ->", important[0])
    important_copy = []
    for i in range(len(important[0])):
        important_copy.append(important[0][i])
    important_copy.append(data.shape[1] - 1)
    select_col = []
    for i in range(len(important_copy)):
        select_col.append(data.columns[important_copy[i]])
    data_selected = pandas.DataFrame(data, columns=select_col)
    data_selected = preprocess.data_normalization(data_selected, have_target=True)
    print("data_selected.shape ->", data_selected.shape)
    print("data_selected.columns ->", data_selected.columns)
    model = decomposition.PCA(n_components=parameters.N_COMPONENTS)
    X = model.fit_transform(data_selected.iloc[:, :-1])
    pos = pandas.DataFrame()
    pos['X'] = X[:, 0]
    pos['Y'] = X[:, 1]
    target = data.columns[data.shape[1] - 1]
    pos[target] = data_selected[target]
    axis = pos[pos[target] == 0].plot(kind='scatter', x='X', y='Y', color='blue', label=0)
    pos[pos[target] == 1].plot(kind='scatter', x='X', y='Y', color='red', label=1, ax=axis)
    print("explained_variance_ratio_ ->", model.fit(data_selected.iloc[:, :-1].values).explained_variance_ratio_)
    plt.title("PCA")
    plt.show()


def FastICA(data, important):
    data = data.sample(frac=parameters.SAMPLE_RATIO).reset_index(drop=True)
    print("data.shape ->", data.shape)
    print("important[0] ->", important[0])
    important_copy = []
    for i in range(len(important[0])):
        important_copy.append(important[0][i])
    important_copy.append(data.shape[1] - 1)
    select_col = []
    for i in range(len(important_copy)):
        select_col.append(data.columns[important_copy[i]])
    data_selected = pandas.DataFrame(data, columns=select_col)
    data_selected = preprocess.data_normalization(data_selected, have_target=True)
    print("data_selected.shape ->", data_selected.shape)
    print("data_selected.columns ->", data_selected.columns)
    model = decomposition.FastICA(n_components=parameters.N_COMPONENTS)
    X = model.fit_transform(data_selected.iloc[:, :-1])
    pos = pandas.DataFrame()
    pos['X'] = X[:, 0]
    pos['Y'] = X[:, 1]
    target = data.columns[data.shape[1] - 1]
    pos[target] = data_selected[target]
    axis = pos[pos[target] == 0].plot(kind='scatter', x='X', y='Y', color='orange', label=0)
    pos[pos[target] == 1].plot(kind='scatter', x='X', y='Y', color='red', label=1, ax=axis)
    plt.title("FastICA")
    plt.show()


def euclidean(data, important):
    data = data.sample(frac=parameters.SAMPLE_RATIO).reset_index(drop=True)
    print("data.shape ->", data.shape)
    print("important[0] ->", important[0])
    important_copy = []
    for i in range(len(important[0])):
        important_copy.append(important[0][i])
    important_copy.append(data.shape[1] - 1)
    select_col = []
    for i in range(len(important_copy)):
        select_col.append(data.columns[important_copy[i]])
    data_selected = pandas.DataFrame(data, columns=select_col)
    data_selected = preprocess.data_normalization(data_selected, have_target=True)
    print("data_selected.shape ->", data_selected.shape)
    print("data_selected.columns ->", data_selected.columns)
    similarities = euclidean_distances(data_selected.iloc[:, :-1].values)
    model = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
    X = model.fit(similarities).embedding_
    pos = pandas.DataFrame(X, columns=['X', 'Y'])
    pos['X'] = X[:, 0]
    pos['Y'] = X[:, 1]
    target = data.columns[data.shape[1] - 1]
    pos[target] = data_selected[target]
    axis = pos[pos[target] == 0].plot(kind='scatter', x='X', y='Y', color='cyan', label=0)
    pos[pos[target] == 1].plot(kind='scatter', x='X', y='Y', color='red', label=1, ax=axis)
    plt.title("euclidean_distances")
    plt.show()


def tSNE(data, important):
    data = data.sample(frac=parameters.SAMPLE_RATIO).reset_index(drop=True)
    print("data.shape ->", data.shape)
    print("important[0] ->", important[0])
    important_copy = []
    for i in range(len(important[0])):
        important_copy.append(important[0][i])
    important_copy.append(data.shape[1] - 1)
    select_col = []
    for i in range(len(important_copy)):
        select_col.append(data.columns[important_copy[i]])
    data_selected = pandas.DataFrame(data, columns=select_col)
    data_selected = preprocess.data_normalization(data_selected, have_target=True)
    print("data_selected.shape ->", data_selected.shape)
    print("data_selected.columns ->", data_selected.columns)
    date_embedded = TSNE(n_components=2).fit_transform(data_selected.iloc[:, :-1])
    pos = pandas.DataFrame(date_embedded, columns=['X', 'Y'])
    target = data.columns[data.shape[1] - 1]
    pos[target] = data_selected[target]
    axis = pos[pos[target] == 0].plot(kind='scatter', x='X', y='Y', color='fuchsia', label=0)
    pos[pos[target] == 1].plot(kind='scatter', x='X', y='Y', color='red', label=1, ax=axis)
    plt.title("tSNE")
    plt.show()


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path,
                                                                                                       test_mode=True)
    # end_off, merge, end_off_feature, merge_feature = \
    #     preprocess.data_cleaning(end_off), preprocess.data_cleaning(merge), \
    #     preprocess.data_cleaning(end_off_feature), preprocess.data_cleaning(merge_feature)
    important, important_h = single_feature_distribution.single_feature(end_off, end_off_feature, end_off_target, False)
    factor_analysis(end_off, important_h)
    PCA(end_off, important_h)
    FastICA(end_off, important_h)
    euclidean(end_off, important_h)
    tSNE(end_off, important_h)
