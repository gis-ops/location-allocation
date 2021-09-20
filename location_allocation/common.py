"""Common classes used across all models"""
import numpy as np


class Result:
    def __init__(self, time_elapsed: int, solution: dict):
        """
        Result class

        :param time_elapsed: the time the solver occupied to compute the result
        :param solution: the solution object
        """
        self.time_elapsed = time_elapsed
        self.solution = solution


class Config:
    def __init__(
        self,
        algorithm_name: str,
        points: np.ndarray,
        facilities: np.ndarray,
        dist_matrix: np.ndarray,
        dist_cutoff: int,
        **kwargs,
    ):
        """
        Constructs the config object for the model.

        :param algorithm_name: class name of the algorithm
        :param points:  Numpy array of shape (n_points, 2).
        :param facilities: Numpy array of shape (n_facilities, 2)
        :param dist_matrix: Numpy array of shape (n_points, n_facilities).
            The distance matrix of points to facilities.
        :param dist_cutoff: Cost cutoff which can be used to exclude points
            from the distance matrix which feature a greater cost.
        :param **kwargs: how many facilities to choose from

        :raises ValueError: if points, dist_matrix, facilities is not ndarray
        :raises ValueError: if facilities_to_site is >= len(facilities)
        """
        # common settings for all algorithms
        self.points = points
        self.facilities = facilities
        self.dist_matrix = dist_matrix
        self.dist_cutoff = dist_cutoff

        # specific settings for algorithms and weights
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

        if algorithm_name in {
            "MaximizeCoverageCapacitated",
            "MaximizeCoverage",
            "MaximizeCoverageMinimizeCost",
        }:
            if self.facilities_to_site >= len(facilities):
                raise ValueError(
                    f"facilities to choose is set to {self.facilities_to_site} but must be lesser equal than facilities {len(facilities)}."
                )
