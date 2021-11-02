# -*- coding: utf-8 -*-

"""
data regression using SVM
"""


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
import numpy as np
import parameters
import load_data
import preprocess
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, NuSVC
import liner_regression_ols
import time
import model_analysis


def cs_svm(end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target):
    model, pca_model = train_cs_svm(merge_feature, merge_target, 3, balance=False)
    model_analysis.model_analysis(end_off_feature, end_off_target, model, end_off, 3, normal=True, pca=True, pca_model=pca_model)


def train_cs_svm(X_train, Y_train, mode=3, balance=True):
    init_time = time.time()
    print("SVM training ...")
    if balance:
        X_train, Y_train = preprocess.un_balance(X_train, Y_train, ratio="minority", mode=2, ensemble=False)
    PCA_model = PCA(n_components=parameters.N_COMPONENTS)
    # X = model.fit_transform(data_selected.iloc[:, :-1])
    X_train = PCA_model.fit_transform(X_train.iloc[:, :-1])
    # param_grid = {
    #     "C": [5e1, 1e2, 5e2],
    #     "gamma": [0.005, 0.01, 0.05]
    # }
    # clf = SVC(kernel="rbf", verbose=1, class_weight="balanced")
    # clf.fit(np.array(X_train), np.array(Y_train.values.ravel()))
    # print("best estimator ->", clf.best_estimator_)
    svc = SVC(kernel="rbf", verbose=1, class_weight="balanced")
    Linear = make_pipeline(StandardScaler(), svc)
    Linear.fit(np.array(X_train), np.array(Y_train))
    print("SVM done, time ->", time.time() - init_time)
    model_analysis.Model_List_1_time[mode] = time.time() - init_time
    return Linear, PCA_model


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path,
                                                                                                       test_mode=False)
    cs_svm(end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target)
