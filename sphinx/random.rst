Random number generation
========================

To cover: (1) seeding a generator, (2) controlling multiple streams, (3) drawing from different distributions

Seeding
-------

* ``dust::random::seed`` (two forms)

Random number state
-------------------

* ``dust::random::xoshiro_state`` and type ``int_type``, static method ``size``
* ``dust::random::next`` (is this public?)
* ``dust::random::random_real``
* ``dust::random::random_int``
* ``dust::random::jump``
* ``dust::random::long_jump``  

Parallel random number object
-----------------------------

* construct ``prng``
* methods - ``size``, ``jump`` (delete?), ``long_jump``, ``state``, ``export_state``, ``import_state``, ``deterministic``

Distributions
-------------

* ``dust::random::binomial``
* ``dust::random::exponential``
* ``dust::random::multinomial``
* ``dust::random::poisson``
* ``dust::random::uniform``

Version
-------

* ``DUST_VERSION_(MAJOR|MINOR|PATCH|STRING|CODE)``
