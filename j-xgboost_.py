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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb
import liner_regression_ols
import time


def XGBoost(end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target):
    end_off, end_off_feature, merge, merge_feature = \
        preprocess.data_normalization(end_off, have_target=True), \
        preprocess.data_normalization(end_off_feature, have_target=False), \
        preprocess.data_normalization(merge, have_target=True), \
        preprocess.data_normalization(merge_feature, have_target=False)
    model = trainXGBoost(merge_feature, merge_target, 1)
    model_analysis.model_analysis(end_off_feature, end_off_target, model, end_off, 1, normal=True)


def trainXGBoost(X_train, Y_train, mode):
    print("XGBoost training...")
    init_time = time.time()
    X_train, Y_train = preprocess.un_balance(X_train, Y_train, ratio="minority", mode=2, ensemble=False)
    xgb.set_config(verbosity=1)
    model = xgb.XGBClassifier(n_estimators=1000, max_depth=100, eval_metric='mlogloss')
    # model = CalibratedClassifierCV(XGB, method='isotonic', cv=2)
    model.fit(np.array(X_train), np.array(Y_train))
    print("XGBoost done, time ->", time.time() - init_time)
    model_analysis.Model_List_1_time[mode] = time.time() - init_time
    return model


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path,
                                                                                                       test_mode=False)
    XGBoost(end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target)
