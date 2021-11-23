Seeding
==========

We can initialise the random number state with any state that is not zero everywhere. Do do this, we take an integer and pass it through the ``splitmix64`` algorithm until we have enough random state; this depends on the generator involved; a xoshiro512 generator requires more starting seed than a xoshiro128 generator.

These seedings functions are designed so that any single integer may be passed into them.  The first form returns an initialised state:

.. doxygenfunction:: seed(uint64_t)

The second initialises a state in-place:

.. doxygenfunction:: seed(T&, uint64_t)

Example
-------

.. literalinclude:: examples/random-seed.cpp

.. literalinclude:: examples/random-seed.out
   :language: text
