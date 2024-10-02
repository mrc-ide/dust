Query version information
=========================

We define several constants that can be used for feature testing and reporting:

.. doxygendefine:: DUST_VERSION_MAJOR
.. doxygendefine:: DUST_VERSION_MINOR
.. doxygendefine:: DUST_VERSION_PATCH
.. doxygendefine:: DUST_VERSION_STRING

The most useful for testing is ``DUST_VERSION_CODE`` which is built from the major, minor and patch numbers (10000 * major + 100 + minor + patch), so version ``1.23.45`` would be ``12345`` and ``0.4.8`` would be ``408``

.. doxygendefine:: DUST_VERSION_CODE

Example
-------

.. literalinclude:: examples/random-version.cpp

.. literalinclude:: examples/random-version.out
   :language: text
