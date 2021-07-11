"""
conftest.py
"""

# https://stackoverflow.com/questions/65116284/python-unittest-debugging-in-vscode
import pytest
import numpy as np


@pytest.fixture
def demand():

    xvalues = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    yvalues = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    xv, yv = np.meshgrid(xvalues, yvalues)
    coordinates = np.dstack((xv, yv)).reshape((np.power(len(xvalues), 2), 2))

    return coordinates


@pytest.fixture
def mclp_facilities():

    coordinates = np.array(
        [
            [-10, 10],
            [10, 10],
            [10, -10],
            [-10, -10],
            [-2.5, 2.5],
            [2.5, 2.5],
            [2.5, -2.5],
            [-2.5, -2.5],
        ]
    )

    return coordinates