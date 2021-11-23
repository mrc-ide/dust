Primitive random number generation functions
============================================

Uniform
-------

The workhorse random number generating function is `random_real`:

.. doxygenfunction:: random_real

Integer
--------

There is also a low-level function for generating integers of a given width:

.. doxygenfunction:: random_int

Note that this **does not** generate an arbitrary integer within some specified range; you are limited to C++ integer types.

Normal
------

The standard normal distribution also gets special treatment.

.. doxygenfunction:: random_normal

Example
-------

.. literalinclude:: examples/random-primitive.cpp

.. literalinclude:: examples/random-primitive.out
   :language: text
