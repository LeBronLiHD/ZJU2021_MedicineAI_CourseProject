# -*- coding: utf-8 -*-

"""
data regression using XGBoost
"""


from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
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
from sklearn.calibration import CalibratedClassifierCV


def XGBoost(data, feature, target, balance):
    print("feature ->", feature.columns)
    print("target ->", target.columns)
    feature = preprocess.data_normalization(feature)
    X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size=0.35, random_state=1)
    if balance:
        X_train, Y_train = preprocess.un_balance(X_train, Y_train, ratio="minority", mode=2, ensemble=False)
    xgb.set_config(verbosity=1)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=10, eval_metric='mlogloss')
    # model = CalibratedClassifierCV(XGB, method='isotonic', cv=2)
    model.fit(np.array(X_train), np.array(Y_train))
    print("model done.")
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    # evaluate model
    scores = cross_val_score(model, np.array(X_test), np.array(Y_test.values.ravel()), scoring="roc_auc", cv=cv, n_jobs=6)
    # summarize performance
    print("mean roc_auc: %.8f" % np.mean(scores))
    Y_pred = model.predict(np.array(X_test))
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
    for i in range(size):
        if Y_pred[i] != 0 and Y_pred[i] != 1:
            print("fuck", end=" ")
    # standard = liner_regression_ols.get_best_divide_line(Y_pred, Y_test, count, size, show_image=False)
    standard = 0.5
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
    liner_regression_ols.plot_pred(data, model, standard, "xgb")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path,
                                                                                                       test_mode=True)
    XGBoost(end_off, end_off_feature, end_off_target, balance=True)
