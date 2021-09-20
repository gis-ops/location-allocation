Welcome to location-allocations's documentation!
================================================

:Documentation: https://location-allocation.readthedocs.io/
:Source Code: https://github.com/gis-ops/location-allocation
:Issue Tracker: https://github.com/gis-ops/location-allocation/issues
:PyPI: https://pypi.org/project/location-allocation


.. toctree::
    :maxdepth: 4
    :caption: Contents

    index


Installation
~~~~~~~~~~~~

::

    pip install location_allocation

Location Allocation Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: location_allocation
   :members:

   .. automethod:: __init__


Configuration Object
----------------------------

.. autoclass:: location_allocation.common.Config
   :members:

   .. automethod:: __init__


Maximize Coverage
-----------------

.. autoclass:: location_allocation.MaximizeCoverage
   :members:

   .. automethod:: __init__

Maximize Capacitated Coverage
-----------------------------

.. autoclass:: location_allocation.MaximizeCoverageCapacitated
   :members:

   .. automethod:: __init__

Maximize Coverage Minimize Cost
-------------------------------
    
.. autoclass:: location_allocation.MaximizeCoverageMinimizeCost
    :members:

    .. automethod:: __init__

Maximize Coverage Minimize Facilities
-------------------------------------

.. autoclass:: location_allocation.MaximizeCoverageMinimizeFacilities
    :members:

    .. automethod:: __init__


Result Object
-------------

.. autoclass:: location_allocation.common.Result
    :members:

    .. automethod:: __init__

#Exceptions
#~~~~~~~~~~

#.. autoclass:: routingpy.exceptions.RouterError
#    :show-inheritance:

Changelog
~~~~~~~~~

See our `Changelog.md`_.

.. _Changelog.md: https://github.com/gis-ops/location-allocation/CHANGELOG.md

Indices and search
==================

* :ref:`genindex`
* :ref:`search`
