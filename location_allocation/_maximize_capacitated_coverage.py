# -*- coding: utf-8 -*-
"""
Maximum capicitated coverage location problem (MCCLP) solved with Mixed Integer Programming.

The result of this computation is a subset of candidate facilities such 
that as many demand points as possible are allocated to these within the cost cutoff value
and considering the capacity of the facility itself.

- Demand points exceeding the facilities cost cutoffs are not considered in the computation
- Demand points within the cost cutoff of one candidate facility has all its weight allocated to it
- If the total demand of a facility is greater than the capacity of the facility, 
only the demand points that maximize total captured demand and minimize total weighted impedance are allocated (double check)
- Demand points within he cost cutoff of two or more facilities has all its demand weight 
allocated to the nearest facility only (double check)
"""

# Author: GIS-OPS UG <enquiry@gis-ops.com>
#
# License: LGPL License

import numpy as np
from mip import *
import time
import random


class RESULT:
    def __init__(self, time_elapsed, solution):
        self.time_elapsed = time_elapsed
        self.solution = solution


class CONFIG:
    def __init__(
        self,
        points,
        facilities,
        facilities_capacities,
        cost_matrix,
        facilities_to_choose,
        cost_cutoff,
    ):
        self.points = points
        self.facilities = facilities
        self.facilities_capacities = facilities_capacities
        self.facilities_to_choose = facilities_to_choose
        self.cost_matrix = cost_matrix
        self.cost_cutoff = cost_cutoff


def generate_initial_solution(D, I, J, C, K):
    """Generate initial solution to use as the starting point for the milp solver.

    Parameters
    ----------
    D : ndarray
        The distance matrix of points to facilities.
    I : int
        number of points I.
    J : int
        number of facilities.
    C : ndarray
        Capacity for each facility of shape (n_facilities_capacities, )
    K : int
        Maximum number of facilities to allocate.

    Returns
    -------
    solution : list
        A list of pairs (i, j) denoting that point i is covered by facility j.
    """

    Is = list(range(0, I))  # list of points
    max_number_of_trials = (
        1000  # if a feasible solution is not found after this many trials,
    )
    # the least infeasible solution is returned
    number_of_trials = 0
    best_solution = []  # least infeasible solution
    best_number_of_assigned_facilities = 0
    while True:
        number_of_trials += 1
        Js = random.sample(list(range(0, J)), K)  # random selection of K facilities
        solution = []

        random.shuffle(Is)
        random.shuffle(Js)
        assigned_facilities = 0
        for j in Js:
            points_assigned_to_facility = 0
            for i in Is:
                if points_assigned_to_facility < C[j] and D[i, j] == 1:
                    points_assigned_to_facility += 1
                    solution.append([i, j])

            if points_assigned_to_facility != 0:
                assigned_facilities += 1

        if assigned_facilities > best_number_of_assigned_facilities:
            best_number_of_assigned_facilities = assigned_facilities
            best_solution = solution

        if assigned_facilities == K or number_of_trials > max_number_of_trials:
            if number_of_trials > max_number_of_trials:
                print("Feasible solution not found, return least infeasible solution")
            else:
                print("Feasible solution found")
            return best_solution


def maximize_capacitated_coverage(
    points,
    facilities,
    facilities_capacities,
    cost_matrix,
    facilities_to_choose,
    cost_cutoff,
    max_seconds=200,
):
    """Solve Maximum capacitated coverage location problem with MIP.

    Given an arbitrary amount of demand [points], find a subset [facilities_to_choose]
    of [facilities] given a certain [capacities] within the [cost_cutoff] and cover as many points as possible.

    Parameters
    ----------
    points : ndarray
        Numpy array of shape (n_points, 2).
    facilities : ndarray
        Numpy array of shape (n_facilities, 2).
    facilities_capacities : ndarray
        Numpy array of shape (n_facilities_capacities, ).
        Must be the same length as facilities with capacities as integers.
    cost_matrix : ndarray
        Numpy array of shape (n_points, n_facilities).
        The distance matrix of points to facilities.
    facilities_to_choose : int
        The amount of facilites to choose, must be lesser equal n_facilities.
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

    """

    mcclp = MAXIMIZE_CAPACITATED_COVERAGE(
        points=points,
        facilities=facilities,
        facilities_capacities=facilities_capacities,
        cost_matrix=cost_matrix,
        facilities_to_choose=facilities_to_choose,
        cost_cutoff=cost_cutoff,
    )
    mcclp.optimize(max_seconds=max_seconds)
    return mcclp.model, mcclp.result


class MAXIMIZE_CAPACITATED_COVERAGE:
    """Solve Maximum capacitated coverage location problem with MIP.

    Given an arbitrary amount of demand [points], find a subset [facilities_to_choose]
    of [facilities] within the [cost_cutoff] and cover as many points as possible.

    Parameters
    ----------
    points : ndarray
        Numpy array of shape (n_points, 2).
    facilities : ndarray
        Numpy array of shape (n_facilities, 2).
    facilities_capacities : ndarray
        Numpy array of shape (n_facilities_capacities, ).
        Must be the same length as facilities with capacities as integers.
    cost_matrix : ndarray
        Numpy array of shape (n_points, n_facilities).
        The distance matrix of points to facilities.
    facilities_to_choose : int
        The amount of facilites to choose, must be lesser equal n_facilities.
    cost_cutoff : int
        Cost cutoff which can be used to exclude points from the distance matrix which
        feature a greater cost.
    max_seconds : int, default=200
        The maximum amount of seconds given to the solver.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial import distance_matrix
    >>> from location_allocation import MAXIMIZE_CAPACITATED_COVERAGE
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
    >>> facilities_capacities = np.array([15, 15, 15, 15, 15, 15, 15, 15])
    >>> cost_matrix = distance_matrix(demand, facilities)
    >>> mcclp = MAXIMIZE_CAPACITATED_COVERAGE(
        demand,
        facilities,
        facilities_capacities,
        cost_matrix,
        facilities_to_choose=2,
        cost_cutoff=4
    )
    >>> mcclp.optimize()
    >>> mcclp.result.solution
    {7: [5, 11, 24, 25, 26, 28, 33, 34, 35, 36, 37, 38, 39, 44, 60], 6: [6, 17, 21, 27, 29, 30, 31, 32, 40, 41, 42, 43, 48, 51, 72]}
    >>> opt_facilities = facilities[list(mcclp.result.solution.keys())]
    >>> opt_facilities
    array([[-2.5, -2.5],
       [ 2.5, -2.5]])
    """

    def __init__(
        self,
        points,
        facilities,
        facilities_capacities,
        cost_matrix,
        facilities_to_choose,
        cost_cutoff,
    ):

        self.config = CONFIG(
            points,
            facilities,
            facilities_capacities,
            cost_matrix,
            facilities_to_choose,
            cost_cutoff,
        )

        I = self.config.points.shape[0]
        J = self.config.facilities.shape[0]
        # D = distance_matrix(points, sites)
        mask1 = self.config.cost_matrix <= self.config.cost_cutoff
        self.config.cost_matrix[mask1] = 1
        self.config.cost_matrix[~mask1] = 0

        self.model = mip.Model()
        x = {}  # is facility at site used
        y = {}  # is point covered by any facility
        z = {}  # is point i covered by facility j

        for i in range(I):
            y[i] = self.model.add_var(var_type=mip.BINARY, name="y%d" % i)
        for j in range(J):
            x[j] = self.model.add_var(var_type=mip.BINARY, name="x%d" % j)

        for i in range(I):
            for j in range(J):
                z[i, j] = self.model.add_var(
                    var_type=mip.BINARY, name="z" + str(i) + "_" + str(j)
                )

        initial_solution = generate_initial_solution(
            self.config.cost_matrix,
            I,
            J,
            self.config.facilities_capacities,
            self.config.facilities_to_choose,
        )

        bigM = 1000000
        epsilon = 0.001

        # exactly K allocated facilities
        self.model.add_constr(
            mip.xsum(x[j] for j in range(J)) == self.config.facilities_to_choose
        )

        # a point cannot be allocated to facility if its not within the facility radius
        for i in range(I):
            for j in np.where(self.config.cost_matrix[i] == 0)[0]:
                self.model.add_constr(z[i, j] == 0)

        # if point is covered, it needs to be covered by at least one facility
        for i in range(I):
            self.model.add_constr(
                mip.xsum(z[i, j] for j in np.where(self.config.cost_matrix[i] == 1)[0])
                >= y[i]
            )

        # the number of points allocated to facility must not exceed facility capacity
        for j in range(J):
            self.model.add_constr(
                mip.xsum(z[i, j] for i in range(I))
                <= self.config.facilities_capacities[j]
            )

        # if at least one point is allocated to facility, the facility must be considered as allocated: for all j \in J sum(z_ij) >= 1 TRUEIFF xj = true
        for j in range(J):
            self.model.add_constr(
                -1 + mip.xsum(z[i, j] for i in range(I)) >= -bigM * (1 - x[j])
            )

        for j in range(J):
            self.model.add_constr(
                -1 + epsilon + mip.xsum(z[i, j] for i in range(I)) <= bigM * x[j]
            )
        self.model.objective = mip.maximize(mip.xsum(y[i] for i in range(I)))
        self.model.start = [(z[i, j], 1.0) for (i, j) in initial_solution]
        self.model.max_mip_gap = 0.1

    def optimize(self, max_seconds=200):
        """Optimize Maximize Capacitated Coverage Problem.

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

        solution = {}
        if (
            self.model.status == mip.OptimizationStatus.FEASIBLE
            or self.model.status == mip.OptimizationStatus.OPTIMAL
        ):
            for v in self.model.vars:
                if v.name[0] == "z" and v.x == 1:
                    site_point = v.name.split("_")
                    point_ix = int(site_point[0][1:])
                    site_ix = int(site_point[1])
                    if site_ix not in solution:
                        solution[site_ix] = []
                    solution[site_ix].append(point_ix)

        self.result = RESULT(float(time.time() - start), solution)
        return self
