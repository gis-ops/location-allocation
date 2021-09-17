# Contribution Guidelines

Thanks for considering to make a contribution to the location-allocation library.

## Table of Contents
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:0 orderedList:0 -->

- [Issues](#issues)
- [Submitting fixes](#submitting-fixes)
	- [Setup](#setup)
	- [Tests](#tests)
	- [Documentation](#documentation)

<!-- /TOC -->

## Issues

- Please only submit actual technical issues

- Please make sure you don't submit a duplicate by browsing open and closed issues first and consult the [CHANGELOG](https://github.com/gis-ops/location-allocation/blob/master/CHANGELOG.md) for already fixed issues

## Submitting fixes

We welcome patches and fixes to existing clients and want to make sure everything goes smoothly for you while submitting a PR.

We use the PSF's [`black`](https://github.com/psf/black) to make sure the code style is consistent, and `flake8` as a linter. 

When contributing, we expect you to:

- close an existing issue. If there is none yet for your fix, please [create one](https://github.com/gis-ops/location-allocation/issues/new).
- write/adapt unit tests and/or mock API tests, depending on the introduced or fixed functionality
- limit the number of commits to a minimum, i.e. responsible use of [`git commit --amend [--no-edit]`](https://www.atlassian.com/git/tutorials/rewriting-history#git-commit--amend)
- use meaningful commit messages
- you can branch off `master` and raise a PR against `master` as well

### Setup

0. Clone and install locally:

```bash
git clone https://gitlab.com/gis-ops/location-allocation.git
cd location-allocation
```

1. Create and activate a new virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install development dependencies:
```bash
# From the root of your git project
poetry install
```

3. Run tests to check if all goes well:
```bash
# From the root of your git project
tox -e py38
#or
pytest --cov=location_allocation -x
```

4. Please install the pre-commit hook, so your code gets auto-formatted and linted before committing it:
```bash
# From the root of your git project
pre-commit install
```

### Documentation

If you add or remove new functionality which is exposed to the user/developer, please make sure to document these in the
docstrings. To build the documentation:

```bash
# From the root of your git project
cd docs
make html
```

The documentation will have been added to `location-allocation/docs/build/html` and you can open `index.html` in your web browser to view the changes. 

We realize that *re-structured text* is not the most user-friendly format, but it is the currently best available
documentation format for Python projects. You'll find lots of copy/paste material in the existing implementations.