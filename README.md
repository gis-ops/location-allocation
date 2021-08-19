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

## Models & Examples

The `location_allocation` modules can be simply imported in your script. Somelö basic examples can be found in the `tests` folder and for more complex examples handling bigger amounts of data can be found in the `examples` folder for all problem types. Generally the problems require the following parameters:

1. Candidate facilities
2. Demand points
3. Cut off value
4. Facilties to choose

Depending on the model additional parameters are required or optional which will be explained in the following paragraphs.

### Maximize Coverage

TEXT

```bash
python3 ./examples/maximize_coverage.py
```

### Maximize Capacitated Coverage

TEXT

```bash
python3 ./examples/maximize_coverage_capacitated.py
```

### Maximize Coverage and Minimize Cost


```bash
python3 ./examples/maximize_coverage_minimize_cost.py
```

### Maximize Coverage and Minimize Facilities

TEXT 

```bash
python3 ./examples/maximize_coverage_minimize_facilities.py
```

## Experiments

To give you an idea about performance we have run a set of experiments for different problem sizes and time allocations for the solver for all models which you can find in the following table. If you want to run these experiments yourself you can find the script to do so in the `examples` folder.

All models.. with
Different problem sizes
Different min_max_gap
Different max_seconds
Result

## QGIS

This module is part of the Open Source Network Analyst plugin for QGIS which you can find [here](https://). If you are interested in recipies how to navigate the plugin please visit our [website](https://networkanalyst.gis-ops.com). 

## Contributing

We appreciate any kind of feature requests or ideas for improvements. If you are planning to raise a pull request we created a (contribution guideline)[https://github.com/gis-ops/location-allocation/blob/master/CONTRIBUTING.md].

