# -*- coding: utf-8 -*-


import os
import parameters


def load_data(path):
    print("Load Data Begin!")
    print("Data Path ->", path)
    for file in os.listdir(path):
        print(file)
    print("Load Data Finished!")


if __name__ == '__main__':
    path = parameters.DATA_PATH
    load_data(path)
