# -*- coding: utf-8 -*-
"""
Maximize coverage minimize facilities location problem solved with Mixed Integer Programming.

Summary: (Una Help)
The result of this computation is a subset of candidate facilities such 
that...


Problem Objective: (Una Help)
...

Notes: (Una help)
- Demand points exceeding the facilities cost cutoffs are not considered in the computation
- Demand points within the cost cutoff of one candidate facility has all its weight allocated to it
- If the total demand of a facility is greater than the capacity of the facility, 
only the demand points that maximize total captured demand and minimize total weighted impedance are allocated (double check)
- Demand points within he cost cutoff of two or more facilities has all its demand weight 
allocated to the nearest facility only (double check)
"""
# Author: GIS-OPS UG <enquiry@gis-ops.com>
#
# License: GPL License

import numpy as np
from mip import *
import time
import random
from .common import CONFIG

logger = logging.getLogger("la")


class RESULT:
    def __init__(self, time_elapsed, solution):
        self.time_elapsed = time_elapsed
        self.solution = solution


def generate_initial_solution(D, I, J, K):
    """Generate initial solution to use as the starting point for the milp solver.
    Input: distance matrix D;
           number of points I;
           number of facilities J;
           max number of facilities to allocate K
    Return:
        solution: a list of pairs (i, j) denoting that point i is covered by facility j.
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
                if D[i, j] == 1:
                    points_assigned_to_facility += 1
                    solution.append([i, j])

            if points_assigned_to_facility != 0:
                assigned_facilities += 1

        if assigned_facilities > best_number_of_assigned_facilities:
            best_number_of_assigned_facilities = assigned_facilities
            best_solution = solution

        if assigned_facilities == K or number_of_trials > max_number_of_trials:
            if number_of_trials > max_number_of_trials:
                logger.debug(
                    "Feasible solution not found, return least infeasible solution"
                )
            else:
                logger.debug("Feasible solution found")
            return best_solution


def maximize_coverage_minimize_facilities(
    points, facilities, cost_matrix, cost_cutoff, max_seconds=200, **kwargs
):
    """Solve Maximum capacitated coverage by minimizing
    facilities location problem with MIP.

    Parameters
    ----------
    points : ndarray
        Numpy array of shape (n_points, 2).
    facilities : ndarray
        Numpy array of shape (n_facilities, 2).
    cost_matrix : ndarray
        Numpy array of shape (n_points, n_facilities).
        The distance matrix of points to facilities.
    cost_cutoff : int
        Cost cutoff which can be used to exclude points from the distance matrix which
        feature a greater cost.
    max_seconds : int, default=200
        The maximum amount of seconds given to the solver.
    max_facilities : int
        The amount of facilites to choose, must be lesser equal n_facilities.
    facility_minimisation_weight : int, default=10
        Una Help
    coverage_maximisation_weight : int, default=1
        Una Help
    total_distance_weight : int, default=1
        Una Help

    Returns
    -------
    model : <mip.model.Model>
        MIP Model class documented here https://docs.python-mip.com/en/latest/classes.html
    result : <location_allocation._maximize_coverage_minimize_facilities.RESULT>
        A result object containing the optimal facilities coordinates (opt_facilities)
        and elapsed time in seconds (time_elapsed).

    Notes
    -----
    For an example, see :ref:`examples/maximize_coverage_minimize_facilities.py
    <sphx_glr_auto_examples_maximize_coverage_minimize_facilities.py>`.

    """

    mcmflp = MAXIMIZE_COVERAGE_MINIMIZE_FACILITIES(
        points=points,
        facilities=facilities,
        cost_matrix=cost_matrix,
        cost_cutoff=cost_cutoff,
        **kwargs
    )
    mcmflp.optimize(max_seconds=max_seconds)
    return mcmflp.model, mcmflp.result


class MAXIMIZE_COVERAGE_MINIMIZE_FACILITIES:
    """Solve Maximum coverage minimize facilities location problem with MIP.

    Parameters
    ----------
    points : ndarray
        Numpy array of shape (n_points, 2).
    facilities : ndarray
        Numpy array of shape (n_facilities, 2).
    cost_matrix : ndarray
        Numpy array of shape (n_points, n_facilities).
        The distance matrix of points to facilities.
    max_facilities : int
        The amount of facilites to choose, must be lesser equal n_facilities.
    cost_cutoff : int
        Cost cutoff which can be used to exclude points from the distance matrix which
        feature a greater cost.
    max_facilities : int
        The amount of facilites to choose, must be lesser equal n_facilities.
    facility_minimisation_weight : int, default=10
        Una Help
    coverage_maximisation_weight : int, default=1
        Una Help
    total_distance_weight : int, default=1
        Una Help
    """

    def __init__(self, points, facilities, cost_matrix, cost_cutoff, **kwargs):

        self.config = CONFIG(
            self.__class__.__name__,
            points,
            facilities,
            cost_matrix,
            cost_cutoff,
            **kwargs
        )

        I = self.config.points.shape[0]
        J = self.config.facilities.shape[0]

        mask1 = self.config.cost_matrix <= self.config.cost_cutoff
        self.config.cost_matrix[mask1] = 1
        self.config.cost_matrix[~mask1] = 0

        self.model = mip.Model()
        # Add variables
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

        initialSolution = generate_initial_solution(
            self.config.cost_matrix, I, J, self.config.max_facilities
        )

        bigM = 1000000
        epsilon = 0.001

        # Add constraints
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

        # if at least one point is allocated to facility, the facility must be considered as allocated: for all j \in J sum(z_ij) >= 1 TRUEIFF xj = true
        for j in range(J):
            self.model.add_constr(
                -1 + mip.xsum(z[i, j] for i in range(I)) >= -bigM * (1 - x[j])
            )

        for j in range(J):
            self.model.add_constr(
                -1 + epsilon + mip.xsum(z[i, j] for i in range(I)) <= bigM * x[j]
            )

        facility_minimisation_weight = self.config.facility_minimisation_weight
        coverage_maximisation_weight = self.config.coverage_maximisation_weight

        # objective: minimize number of used facilities while maximizing the coverage
        self.model.objective = mip.minimize(
            facility_minimisation_weight * mip.xsum(x[j] for j in range(J))
            - coverage_maximisation_weight * mip.xsum(y[i] for i in range(I))
        )

        self.model.start = [(z[i, j], 1.0) for (i, j) in initialSolution]

        # Ask Una
        self.model.max_mip_gap = self.config.max_gap

    def optimize(self, max_seconds=200):
        """Optimize Maximize Coverage Minimize Cost Problem.

        Parameters
        ----------
        max_seconds : int
            The maximum amount of seconds given to the solver.

        Returns
        -------
        self : object
            Returns an instance of self consisting of the configuration,
            mip model and points to facility allocations.
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
