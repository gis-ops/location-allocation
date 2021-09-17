# Location Allocation with Mixed Integer Programming

Add Image

This repository features a set of location-allocation algorithms to determine an optimal location for one or more facilities given a set of candidate facilities as well as a given set of demand points. Depending on the nature of the algorithm demand points are assigned to one or more facilities, taking into account factors such as the number of facilities available, their capacity, and the maximum cost from a facility to a point. Typical examples evolve around emergency response services and where to locate facilities so that the greatest number of people in a region can be reached within a given time.

## Installation

### Install via poetry:

`poetry add location-allocation`

### Install using pip with

`pip install location-allocation`

### Or the lastest from source

`pip install git+git://github.com/gis-ops/location-allocation`

## Problem Types

The `location_allocation` modules can be simply imported. Some basic examples can be found in the `tests` folder and for more complex examples handling larger amounts of data can be found in the `examples` folder as jupyter notebooks. Generally the location allocation problems require as input candidate facilities, demand points, a cost matrix, a cut off value and facilties to site. This repository features 4 different types of problems.

- Maximize Coverage: This problem is used to site facilities from candidate facilites covering the largest amount of demand points within a cost cutoff value.

- Maximize Capacitated Coverage: This problem is used to site facilities from candidate capacity bound facilites covering the largest amount of demand points within a cost cutoff value.

- Maximize Coverage and Minimize Cost: This problem is used to site facilities from candidate facilites covering the largest amount of demand points and minimizing the overall cost within a cost cutoff value.

- Maximize Coverage and Minimize Facilities: This problem is used to site facilities from candidate facilites covering the largest amount of demand points within a cost cutoff value. The difference to maximize coverage is that the number of facilities to be sited does not have to be specified.


## Performance

To give you an idea about performance we have run a set of experiments for different problem sizes. If you want to run these experiments yourself you can find the jupyter notebook in the `examples` folder.



## Contributing

We appreciate any kind of feature requests or ideas for improvements. If you are planning to raise a pull request we created a (contribution guideline)[https://github.com/gis-ops/location-allocation/blob/master/CONTRIBUTING.md].

