Random number interface from R
==============================

We provide several functions for use from within R packages, all of which are in the namespace ``dust::random::r``. These are further described in `the package vignette <https://mrc-ide.github.io/dust/articles/rng_package.html>`__

To access these functions you must have the ``cpp11`` package available and include ``<dust/r/random.hpp>``.

Seeding the generator
---------------------

As a slightly more convienient interace for seeding generators

.. doxygenfunction:: as_rng_seed

Once you have a seed, there's no direct interaction with the dust R API any further - you have a seed and can draw numbers as described elsewhere in this manual.

Persistent streams
------------------

To provide access to `persistent streams <https://mrc-ide.github.io/dust/articles/rng_package.html#basic-implementation-using-dust>`__ (see also `the reference documentation <https://mrc-ide.github.io/dust/reference/dust_rng_pointer.html>`__)

.. doxygenfunction:: rng_pointer_get
