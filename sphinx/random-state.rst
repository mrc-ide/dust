Random number state
===================

Typically, you don't need to think much about the state used; in the simple example we do

.. literalinclude:: examples/random-simple.cpp

This pattern will typically be sufficient - use `dust::random::generator` to select an appropriate algorithm for your floating point type, and use `auto` to save it.  This will select `dust::random::xoshiro256plus` for `double` and `dust::random::xoshiro128plus` for `float`

Core state class
----------------

Every random number state is built on the same core class

.. doxygenclass:: dust::random::xoshiro_state
   :members:

.. note this formats incorrectly, see
   https://github.com/michaeljones/breathe/issues/761

Every random number state uses one of the three scrambler options, described based on the mathematical operations they perform when creating the final integer from the generator

.. doxygenenum:: dust::random::scrambler


Concrete types
--------------

We then define 12 user-usable types:

**32-bit generators**, suitable for generating `float` values

.. doxygentypedef:: xoshiro128starstar
.. doxygentypedef:: xoshiro128plusplus
.. doxygentypedef:: xoshiro128plus

**64 bit generators**, suitable for generating either `double` or `float` values, but differing in the size of the internal state:

*128 bits*:

.. doxygentypedef:: xoroshiro128starstar
.. doxygentypedef:: xoroshiro128plusplus
.. doxygentypedef:: xoroshiro128plus

*256 bits*:

.. doxygentypedef:: xoshiro256starstar
.. doxygentypedef:: xoshiro256plusplus
.. doxygentypedef:: xoshiro256plus

*512 bits*:

.. doxygentypedef:: xoshiro512starstar
.. doxygentypedef:: xoshiro512plusplus
.. doxygentypedef:: xoshiro512plus

Example
-------

.. literalinclude:: examples/random-state.cpp

.. literalinclude:: examples/random-state.out
   :language: text
