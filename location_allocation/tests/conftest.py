"""
conftest.py
"""

# https://stackoverflow.com/questions/65116284/python-unittest-debugging-in-vscode
import pytest
import numpy as np
from matplotlib import pyplot as plt


@pytest.fixture
def demand():
    """ Creates dummy demand data with sparse and dense areas """

    sparse_xvalues = np.arange(start=-1, stop=1.1, step=1)
    sparse_yvalues = np.arange(start=-1, stop=1.1, step=1)
    sparse_xv, sparse_yv = np.meshgrid(sparse_xvalues, sparse_yvalues)
    sparse_coordinates = np.dstack((sparse_xv, sparse_yv)).reshape(
        (np.power(len(sparse_xvalues), 2), 2)
    )

    dense_top_xvalues = np.arange(start=-1, stop=1.1, step=0.2)
    dense_top_yvalues = np.arange(start=1.2, stop=2.1, step=0.2)
    dense_top_xv, dense_top_yv = np.meshgrid(dense_top_xvalues, dense_top_yvalues)
    dense_top_coordinates = np.dstack((dense_top_xv, dense_top_yv)).reshape(
        (len(dense_top_xvalues) * len(dense_top_yvalues), 2)
    )

    dense_bottom_xvalues = np.arange(start=-1, stop=1.1, step=0.2)
    dense_bottom_yvalues = np.arange(start=-2, stop=-1.1, step=0.2)
    dense_bottom_xv, dense_bottom_yv = np.meshgrid(
        dense_bottom_xvalues, dense_bottom_yvalues
    )
    dense_bottom_coordinates = np.dstack((dense_bottom_xv, dense_bottom_yv)).reshape(
        (len(dense_bottom_xvalues) * len(dense_bottom_yvalues), 2)
    )

    coordinates = np.concatenate(
        [sparse_coordinates, dense_top_coordinates, dense_bottom_coordinates]
    )
    # x, y = coordinates.T
    # plt.plot(x, y, marker=".", color="k", linestyle="none")
    # plt.show()

    return coordinates


@pytest.fixture
def mclp_facilities():
    """ Creates dummy faciltilies data """

    coordinates = np.array([[-1, 2], [1, 2], [1, 0], [-1, 0]])

    return coordinates