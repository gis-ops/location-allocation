"""Common classes used across all models"""
import numpy as np


class RESULT:
    def __init__(self, time_elapsed, solution):
        """
        Result class

        :param time_elapsed: the time the solver occupied to compute the result
        :type time_elapsed: int
        :param solution: the solution object
        :type solution: object
        """
        self.time_elapsed = time_elapsed
        self.solution = solution


class CONFIG:
    def __init__(
        self, algorithm_name, points, facilities, cost_matrix, cost_cutoff, **kwargs
    ):
        """
        Constructs the config object for the model

        :param algorithm_name: class name of the algorithm
        :type algorithm_name: str
        :param points:  Numpy array of shape (n_points, 2).
        :type points: ndarray
        :param facilities: Numpy array of shape (n_facilities, 2).
        :type facilities: ndarray
        :param cost_matrix: Numpy array of shape (n_points, n_facilities).
            The distance matrix of points to facilities.
        :type cost_matrix: ndarray
        :param cost_cutoff: Cost cutoff which can be used to exclude points
            from the distance matrix which feature a greater cost.
        :type cost_cutoff: int
        :param **kwargs: Additional parameters for the specific model
        :type **kwargs: dict

        :raises ValueError: if points is not ndarray
        :raises ValueError: if facilities is not ndarray
        :raises ValueError: if cost_matrix is not ndarray
        :raises ValueError: if facilities_to_choose is >= len(facilities)
        """
        # common settings for all algorithms
        if type(points) is np.ndarray:
            self.points = points
        else:
            raise ValueError(f"points is of {type(points)}, should be numpy.ndarray.")
        if type(facilities) is np.ndarray:
            self.facilities = facilities
        else:
            raise ValueError(
                f"facilities is of {type(facilities)}, should be numpy.ndarray."
            )
        if type(cost_matrix) is np.ndarray:
            self.cost_matrix = cost_matrix
        else:
            raise ValueError(
                f"cost_matrix is of {type(cost_matrix)}, should be numpy.ndarray."
            )

        self.cost_cutoff = cost_cutoff

        # specific settings for algorithms and weights
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

        if algorithm_name in {
            "MAXIMIZE_COVERAGE_CAPACITATED",
            "MAXIMIZE_COVERAGE",
            "MAXIMIZE_COVERAGE_MINIMIZE_COST",
        }:
            if self.facilities_to_choose >= len(facilities):
                raise ValueError(
                    f"facilities to choose is set to {self.facilities_to_choose} but must be lesser equal than facilities {len(facilities)}."
                )
