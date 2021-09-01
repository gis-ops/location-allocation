import numpy as np
import pytest


@pytest.fixture
def demand():
    """ Creates dummy demand data with sparse and dense areas """

    sparse_xvalues = np.arange(start=-5, stop=6, step=5)
    sparse_yvalues = np.arange(start=-5, stop=6, step=5)
    sparse_xv, sparse_yv = np.meshgrid(sparse_xvalues, sparse_yvalues)
    sparse_coordinates = np.dstack((sparse_xv, sparse_yv)).reshape((np.power(len(sparse_xvalues), 2), 2))

    dense_top_xvalues = np.arange(start=-5, stop=6, step=1)
    dense_top_yvalues = np.arange(start=6, stop=11, step=1)
    dense_top_xv, dense_top_yv = np.meshgrid(dense_top_xvalues, dense_top_yvalues)
    dense_top_coordinates = np.dstack((dense_top_xv, dense_top_yv)).reshape(
        (len(dense_top_xvalues) * len(dense_top_yvalues), 2)
    )

    dense_bottom_xvalues = np.arange(start=-5, stop=6, step=1)
    dense_bottom_yvalues = np.arange(start=-10, stop=-5, step=1)
    dense_bottom_xv, dense_bottom_yv = np.meshgrid(dense_bottom_xvalues, dense_bottom_yvalues)
    dense_bottom_coordinates = np.dstack((dense_bottom_xv, dense_bottom_yv)).reshape(
        (len(dense_bottom_xvalues) * len(dense_bottom_yvalues), 2)
    )

    demand = np.concatenate([sparse_coordinates, dense_top_coordinates, dense_bottom_coordinates])
    # x, y = demand.T
    # plt.plot(x, y, marker=".", color="k", linestyle="none")
    # plt.show()

    return demand


@pytest.fixture
def facilities():
    """
    Creates dummy faciltilies data
    """

    facilities = np.array([[-5, 10], [5, 10], [5, 0], [-5, 0]])

    return facilities


@pytest.fixture
def capacities():
    """
    Creates dummy faciltilies capacities data
    """
    capacities = np.array([15, 15, 5, 5])

    return capacities
