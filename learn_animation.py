# -*- coding: utf-8 -*-

"""
learn matplotlib.animation
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main():

    fig, ax = plt.subplots()
    x, y = [], []
    line, = plt.plot([], [], '.-')

    def init():
        ax.set_xlim(-10, 10)
        ax.set_ylim(-5, 5)
        return line

    def update(step):
        x.append(step)
        y.append(np.cos(step/3)+np.sin(step**2))
        line.set_data(x, y)
        return line

    ani = FuncAnimation(fig, update, frames=np.linspace(-10, 10, num=100), init_func=init, interval=20)
    # ani.save("learn_animation.png")
    plt.show()


if __name__ == '__main__':
    main()
