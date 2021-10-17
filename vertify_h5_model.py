# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array
import load_data
import filter
import sys
import parameters

sys.dont_write_bytecode = True


def vertify_model(test, expect, total=10):
    # sort by last modified time
    model_lists = os.listdir(parameters.MODEL_SAVE)
    model_lists = sorted(model_lists,
                         key=lambda files: os.path.getmtime(os.path.join(parameters.MODEL_SAVE, files)),
                         reverse=False)
    model_path_vertify = ""
    for modelLists in os.listdir(parameters.MODEL_SAVE):
        model_path_vertify = os.path.join(parameters.MODEL_SAVE, modelLists)
        print(model_path_vertify)

    if model_path_vertify == "":  # if the pwd is NULL
        print("No model saved!")
        exit()

    model = load_model(model_path_vertify)
    print("model loaded!")

    count_vertify = 0
    for i in range(total):
        data_piece = test[i]
        data_piece = np.expand_dims(data_piece, 0)  # 扩展至四维
        output = model.predict(data_piece)
        print("output ->", output)
        print("count ->", i, "   \t-> ", output.argmax())
        plt.figure("Test Image")
        for_show = []
        for i in range(parameters.COLUMNS):
            for_show_unit = []
            for j in range(parameters.COLUMNS):
                for_show_unit.append(data_piece[0][i][j][0])
            for_show.append(for_show_unit)
        print("for_show.shape ->", np.shape(for_show))
        plt.imshow(for_show)
        plt.axis("off")  # 关掉坐标轴为 off
        count_vertify += 1
        title = "CNN fuck " + str(expect[i][0]) + str(expect[i][1]) + str(expect[i][2])
        plt.title(title)  # 图像题目
        plt.show()
