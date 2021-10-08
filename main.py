# -*- coding: utf-8 -*-

"""
main file of the project
"""

import load_data
import parameters


def main():
    path = parameters.DATA_PATH
    data_end_all, data_merge = load_data.load_data(path)


if __name__ == '__main__':
    main()
