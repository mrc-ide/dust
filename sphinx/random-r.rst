Random number interface from R
==============================

We provide several api functions here:

* ``dust::random::r::as_rng_seed``

Once you have a seed, there's no direct interaction with the R API any further - you have a seed and can draw numbers (give example)

These three go together

* ``dust::random::r::rng_pointer_init``
* ``dust::random::r::rng_pointer_get``
* ``dust::random::r::rng_pointer_sync``
