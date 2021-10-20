# -*- coding: utf-8 -*-

"""
data regression using XGBoost
"""


from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

import model_analysis
import parameters
import load_data
import warnings
import preprocess
from sklearn.decomposition import PCA
import xgboost as xgb
import liner_regression_ols
import time


def XGBoost(end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target):
    end_off, end_off_feature, merge, merge_feature = \
        preprocess.data_normalization(end_off, have_target=True), \
        preprocess.data_normalization(end_off_feature, have_target=False), \
        preprocess.data_normalization(merge, have_target=True), \
        preprocess.data_normalization(merge_feature, have_target=False)
    model, pca_model = trainXGBoost(merge_feature, merge_target, 1, pca=True, n_com=30)
    model_analysis.model_analysis(end_off_feature, end_off_target, model, end_off, 1, normal=True, pca=True, pca_model=pca_model)


def trainXGBoost(X_train, Y_train, mode, pca=False, n_com=parameters.N_COMPONENTS):
    print("XGBoost training...")
    init_time = time.time()
    X_train, Y_train = preprocess.un_balance(X_train, Y_train, ratio="minority", mode=2, ensemble=False)
    xgb.set_config(verbosity=1)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=60, eval_metric='mlogloss')
    # model = CalibratedClassifierCV(XGB, method='isotonic', cv=2)
    PCA_model = None
    if pca:
        PCA_model = PCA(n_components=n_com)
        X_train = PCA_model.fit_transform(X_train.iloc[:, :-1])
    model.fit(np.array(X_train), np.array(Y_train))
    print("XGBoost done, time ->", time.time() - init_time)
    model_analysis.Model_List_1_time[mode] = time.time() - init_time
    return model, PCA_model


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path,
                                                                                                       test_mode=False)
    XGBoost(end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target)
