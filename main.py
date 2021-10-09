# -*- coding: utf-8 -*-

"""
main file of the project
"""

import load_data
import parameters
import preprocess


def main():
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path)
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = \
        preprocess.data_cleaning(end_off), preprocess.data_cleaning(merge), preprocess.data_cleaning(end_off_feature), \
        preprocess.data_cleaning(merge_feature), preprocess.data_cleaning(end_off_target), preprocess.data_cleaning(merge_target)


if __name__ == '__main__':
    main()
