# -*- coding: utf-8 -*-
"""
Maximum coverage location problem (MCLP) solved with Mixed Integer Programming.
Inspired by https://github.com/cyang-kth/maximum-coverage-location
"""

# Author: Can Yang <cyang@kth.se>
#         GIS-OPS UG <enquiry@gis-ops.com>
#
# License: MIT License

import numpy as np
from mip import *
import time


class RESULT:
    def __init__(self, time_elapsed, opt_facilities, opt_facilities_indexes):
        self.time_elapsed = time_elapsed
        self.opt_facilities = opt_facilities
        self.opt_facilities_indexes = opt_facilities_indexes


class CONFIG:
    def __init__(
        self, points, facilities, cost_matrix, facilities_to_choose, cost_cutoff
    ):
        self.points = points
        self.facilities = facilities
        self.facilities_to_choose = facilities_to_choose
        self.cost_matrix = cost_matrix
        self.cost_cutoff = cost_cutoff


def maximize_capacitated_coverage(
    points,
    facilities,
    cost_matrix,
    facilities_to_choose,
    cost_cutoff,
    max_seconds=200,
):
    """Solve Maximum coverage location problem with MIP described in Church 1974.

    Given an arbitrary amount of demand [points], find a subset [facilities_to_choose]
    of [facilities] within the [cost_cutoff] and cover as many points as possible.

    Parameters
    ----------

    points : ndarray
        Numpy array of shape (points, 2).
    facilities : ndarray
        Numpy array of shape (facilities, 2).
    cost_matrix : ndarray
        Numpy array of shape (points, facilities, 2).
        The distance matrix of points to facilities.
    facilities_to_choose : int
        The amount of facilites to choose, must be lesser equal len(facilities).
    cost_cutoff : int
        Cost cutoff which can be used to exclude points from the distance matrix which
        feature a greater cost.
    max_seconds : int, default=200
        The maximum amount of seconds given to the solver.


    Returns
    -------
    model : <mip.model.Model>
        MIP Model class documented here https://docs.python-mip.com/en/latest/classes.html
    result : <location_allocation._maximize_coverage.RESULT>
        A result object containing the optimal facilities coordinates (opt_facilities)
        and elapsed time in seconds (time_elapsed).


    Notes
    -----
    For an example, see :ref:`examples/maximize_coverage.py
    <sphx_glr_auto_examples_maximize_coverage.py>`.

    References
    ----------
    Church, R. and C. ReVelle, "The maximal covering location problem".
    In: Papers of the Regional Science Association, pp. 101-118. 1974.
    """

    mclp = MAXIMIZE_CAPACITATED_COVERAGE(
        points=points,
        facilities=facilities,
        cost_matrix=cost_matrix,
        facilities_to_choose=facilities_to_choose,
        cost_cutoff=cost_cutoff,
    )
    mclp.optimize(max_seconds=max_seconds)
    return mclp.model, mclp.result


class MAXIMIZE_CAPACITATED_COVERAGE:
    """Solve Maximum coverage location problem with MIP described in Church 1974.

    Given an arbitrary amount of demand [points], find a subset [facilities_to_choose]
    of [facilities] within the [cost_cutoff] and cover as many points as possible.

    Parameters
    ----------
    points : ndarray
        Numpy array of shape (points, 2).
    facilities : ndarray
        Numpy array of shape (facilities, 2).
    cost_matrix : ndarray
        Numpy array of shape (points, facilities, 2).
        The distance matrix of points to facilities.
    facilities_to_choose : int
        The amount of facilites to choose, must be lesser equal len(facilities)
        to produce results.
    cost_cutoff : int
        Cost cutoff which can be used to exclude points from the distance matrix which
        feature a greater cost.
    max_seconds : int, default=200
        The maximum amount of seconds given to the solver.


    Returns
    -------
    model : <mip.model.Model>
        MIP Model class documented here https://docs.python-mip.com/en/latest/classes.html
    result : <location_allocation._maximize_coverage.RESULT>
        A result object containing the optimal facilities coordinates (opt_facilities)
        and elapsed time in seconds (time_elapsed).


    Notes
    -----
    For an example, see :ref:`examples/maximize_coverage.py
    <sphx_glr_auto_examples_maximize_coverage.py>`.

    References
    ----------
    Church, R. and C. ReVelle, "The maximal covering location problem".
    In: Papers of the Regional Science Association, pp. 101-118. 1974.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial import distance_matrix
    >>> from location_allocation import MAXIMIZE_COVERAGE
    >>> xvalues = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    >>> yvalues = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    >>> xv, yv = np.meshgrid(xvalues, yvalues)
    >>> demand = np.dstack((xv, yv)).reshape((np.power(len(xvalues), 2), 2))
    >>> facilities = np.array(
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
    >>> cost_matrix = distance_matrix(demand, facilities)
    >>> mclp = MAXIMIZE_COVERAGE(
        demand, facilities, cost_matrix, facilities_to_choose=2, cost_cutoff=2
    )
    >>> mclp.optimize()
    >>> mclp.result.opt_facilities
    array([[ 2.5,  2.5],
       [ 2.5, -2.5]])
    """

    def __init__(
        self,
        points,
        facilities,
        cost_matrix,
        facilities_to_choose,
        cost_cutoff,
    ):

        self.config = CONFIG(
            points, facilities, cost_matrix, facilities_to_choose, cost_cutoff
        )

        I = self.config.points.shape[0]
        J = self.config.facilities.shape[0]

        mask1 = self.config.cost_matrix <= self.config.cost_cutoff
        self.config.cost_matrix[mask1] = 1
        self.config.cost_matrix[~mask1] = 0

        self.model = mip.Model()

        x = {}
        y = {}
        for i in range(I):
            y[i] = self.model.add_var(var_type=mip.BINARY, name="y%d" % i)
        for j in range(J):
            x[j] = self.model.add_var(var_type=mip.BINARY, name="x%d" % j)

        self.model.add_constr(
            mip.xsum(x[j] for j in range(J)) == self.config.facilities_to_choose
        )

        for i in range(I):
            self.model.add_constr(
                mip.xsum(x[j] for j in np.where(self.config.cost_matrix[i] == 1)[0])
                >= y[i]
            )

        self.model.objective = mip.maximize(mip.xsum(y[i] for i in range(I)))

    def optimize(self, max_seconds=200):
        """Optimize MCLP.

        Parameters
        ----------
        max_seconds : int
            The maximum amount of seconds given to the solver.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        start = time.time()
        self.model.optimize(max_seconds=max_seconds)
        solution = []
        if (
            self.model.status == mip.OptimizationStatus.FEASIBLE
            or self.model.status == mip.OptimizationStatus.OPTIMAL
        ):
            for v in self.model.vars:
                if v.x == 1 and v.name[0] == "x":
                    solution.append(int(v.name[1:]))
        self.result = RESULT(
            float(time.time() - start), self.config.facilities[solution], solution
        )
        return self
