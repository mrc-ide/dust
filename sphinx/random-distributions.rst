Distributions
=============

Our distribution support will grow over time. If you need a distribution that is not yet supported please `post an issue <https://github.com/mrc-ide/dust/issues/>`.


Uniform
-------

.. doxygenfunction:: dust::random::uniform

Normal
------

.. doxygenfunction:: dust::random::normal

The enum ``dust::random::algorithm::normal`` controls the normal algorithm used for a given draw

.. doxygenenum:: dust::random::algorithm::normal

Binomial
--------

.. doxygenfunction:: dust::random::binomial

Hypergeometric
--------

.. doxygenfunction:: dust::random::hypergeometric

Exponential
-----------

.. doxygenfunction:: dust::random::exponential

Poisson
-------

.. doxygenfunction:: dust::random::poisson

Multinomial
-----------

.. doxygenfunction:: dust::random::multinomial(rng_state_type&, int, const T&, int, U&)

Example
-------

.. literalinclude:: examples/random-univariate.cpp

.. literalinclude:: examples/random-univariate.out
   :language: text
