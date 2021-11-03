# -*- coding: utf-8 -*-

"""
main file of the project
1. analysis and judge
2. predict and prepare
"""

import f_load_data
import f_parameters
import f_preprocess
import warnings
import j_CNN_binary_classification
import j_NN_binary_classification
import j_svm
import j_xgboost
import j_randomforest
import j_decision_tree
import j_factor_pca_ica_euclidean_tsne
import r_ols
import r_pls
import r_bayesian
import r_elasticnet
import r_naive_bayes
import r_RNN_binary_classification

def main():
    warnings.filterwarnings("ignore")
    path = f_parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = \
        f_load_data.f_load_data(path, test_mode=False)


if __name__ == '__main__':
    main()
