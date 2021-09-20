# Location Allocation with Mixed Integer Programming

![Maximize Coverage Example](/examples/img/maximize_coverage_example.png?raw=true "Maximize Coverage Example Problem")

This repository features a set of location-allocation algorithms to determine an optimal location for one or more facilities given a set of candidate facilities as well as a given set of demand points. Depending on the nature of the algorithm demand points are assigned to facilities, taking into account factors such as the number of facilities available, their capacity, and the maximum cost from a facility to a point. Typical examples evolve around emergency response services and where to locate facilities so that the greatest number of people in a region can be reached within a given time or distance. 

## Installation

### Install via poetry:

`poetry add location-allocation`

### Install using pip with

`pip install location-allocation`

### Or the lastest from source

`pip install git+git://github.com/gis-ops/location-allocation`

**location-allocation** is tested against CPython versions 3.7, 3.8, 3.9 and against PyPy3. 


## Documentation
 
The full location-allocation documentation is available at
https://location-allocation.readthedocs.io/en/latest/?badge=latest

A PDF version is also available:
https://location-allocation.readthedocs.io/_/downloads/en/latest/pdf/

## Problem Types and Examples

The `location_allocation` modules can be simply imported. Some basic examples can be found in the `tests` folder and for more complex examples handling larger amounts of data can be found in the `examples` folder as jupyter notebooks. Generally the location allocation problems require as input candidate facilities, demand points, a cost matrix, a cut off value and facilties to site. This repository features 4 different types of problems. It must be noted that each demand point represents a weight of 1.

- [Maximize Coverage](/examples/maximize_coverage.ipynb): This problem is used to site facilities from candidate facilites covering the largest amount of demand points within a cost cutoff value.

- [Maximize Capacitated Coverage](/examples/maximize_coverage_capacitated.ipynb): This problem is used to site facilities from candidate capacity bound facilites covering the largest amount of demand points within a cost cutoff value.

- [Maximize Coverage and Minimize Cost](/examples/maximize_coverage_minimize_cost.ipynb): This problem is used to site facilities from candidate facilites covering the largest amount of demand points and minimizing the overall cost within a cost cutoff value.

- [Maximize Coverage and Minimize Facilities](/examples/maximize_coverage_minimize_facilities.ipynb): This problem is used to site facilities from candidate facilites covering the largest amount of demand points within a cost cutoff value. The difference to maximize coverage is that the number of facilities to be sited does not have to be specified.


## Performance

To give you an idea about performance we have run a set of experiments for different problem sizes. If you want to run these experiments yourself you can find the [jupyter notebook here](/examples/performance.ipynb).

![Maximize Coverage Performance](/examples/img/maximize_coverage.png?raw=true "Maximize Coverage")

![Maximize Capacitated Coverage Performance](/examples/img/maximize_coverage_capacitated.png?raw=true "Maximize Capacitated Coverage")

![Maximize Coverage Minimize Cost Performance](/examples/img/maximize_coverage_minimize_cost.png?raw=true "Maximize Coverage Minimize Cost")

![Maximize Coverage Minimize Facilities Performance](/examples/img/maximize_coverage_minimize_facilities.png?raw=true "Maximize Coverage Minimize Facilities")

## Build status

[![Github Actions Status](https://github.com/gis-ops/location-allocation/workflows/tests/badge.svg?branch=master)](https://github.com/gis-ops/location-allocation/actions)
[![Current version](https://badge.fury.io/gh/gis-ops%2Flocation-allocation.svg)](https://github.com/gis-ops/location-allocation/releases)
[![License](https://img.shields.io/badge/license-GPL-blue.svg)](https://github.com/gis-ops/location-allocation/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/location-allocation/badge/?version=latest)](https://location-allocation.readthedocs.io/en/latest/?badge=latest)
[![MyBinder.org](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gis-ops/location-allocation/master?filepath=examples)


## Contributing

We appreciate any kind of feature requests or ideas for improvements. If you are planning to raise a pull request we created a [contribution guideline](https://github.com/gis-ops/location-allocation/blob/master/CONTRIBUTING.md).

