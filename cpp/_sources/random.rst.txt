Random number generation
========================

To generate random numbers we need to

1. include ``<dust/random/random.hpp>``
2. decide on the generator algorithm to use
3. seed a random number state
4. draw numbers using some distribution function

A simple complete example looks like

.. literalinclude:: examples/random-simple.cpp

which produces output:

.. literalinclude:: examples/random-simple.out
   :language: text

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   random-state
   random-seed
   random-primitive
   random-distributions
   random-jump
   random-parallel
   random-version
   random-r
