# -*- coding: utf-8 -*-

"""
main file of the project
1. analysis and judge
2. predict and prepare
"""

import load_data
import parameters
import preprocess
import warnings


def main():
    warnings.filterwarnings("ignore")
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path,
                                                                                                       test_mode=False)


if __name__ == '__main__':
    main()
