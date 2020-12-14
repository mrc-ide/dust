# dust 0.5.6

* `dust::dust_simulate()` can return the entire model end state (#119)

# dust 0.5.4

* New `$set_data()` method, similar to `reset` but changing only the data/parameters mid-model run, holding state and everything else identical (#114)

# dust 0.5.3

* Add support for configuring dust generation using C++ pseudo-attributes.

# dust 0.5.2

* Back out the interleaved rng state from 0.5.0, which is causing a performance regression

# dust 0.5.1

* Remove prototype GPU interface, in preparation for a new version (#109)

# dust 0.5.0

* The rng objects (`dust_rng`) also get a `$set_state()` method (primarily of debugging interest)

# dust 0.4.12

* New `$set_rng_state()` method, the inverse to `$rng_state()` for taking a previously saved state and setting it into a model (#??)

# dust 0.4.11

* `dust::dust_iterate` now copies names from the index as rownames (#101)

# dust 0.4.9

* The "low `n * p`" branch of the binomial distribution now uses a slightly faster algorithm (#91)

# dust 0.4.8

* `dust::dust_openmp_support()`, `dust::dust_openmp_threads()` and a method `$has_openmp()` on `dust` objects to make determining OpenMP support easier (#97)

# dust 0.4.7

* `dust::dust_package()` validates that the package contains a suitable `src/Makevars` for use with openmp, or creates one if missing (#95)

# dust 0.4.6

* Some examples are now compiled into the package and available via `dust::dust_example()`, reducing the need for a compiler for exploration and making examples and testing faster (#89)

# dust 0.4.5

* Change of behaviour for `seed`, which now seeds from R's random number if given the default value of `NULL` (#85, #87)
* The `rng_state` method can optionally advance the random number stream, making it more suitable for use with `dust::dust_simulate`
* A new utility `dust::dust_rng_long_jump` which can advance the saved state of a dust RNG, suitable for creating independent streams from one saved state.

# dust 0.4.4

* Helper function `dust::dust_simulate` is renamed to `dust::dust_iterate`
* New function `dust::dust_simulate` which simulates many parameter sets and starting points at once, possibly in parallel (#84)

# dust 0.4.3

* If `$set_index()` uses a named index vector, then those names are copied back as rownames on the returned matrix. Similarly, if `$state()` is used with a named index then those names are used as rownames (#81)

# dust 0.4.0

* Can now generate dust objects that run on the GPU (#69)

# dust 0.3.0

* Overhaul of the RNG interface, where we now pass state and use free functions; this is needed to support future development work (#60)

# dust 0.2.2

* New `$rng_state()` method for getting the RNG state as a raw vector

# dust 0.2.0

* Use cpp11 as the backend (#22)

# dust 0.1.5

* Simpler RNG interface; we now always use as many RNG streams as there are particles (#51)

# dust 0.1.4

* New function `dust::dust_simulate` which provides a helper for running a simulation while collecting output (#7)

# dust 0.1.3

* Allow `$set_state()` to accept an initial state of `NULL`, in which case only the time is set (#48)

# dust 0.1.2

* Allow `$set_state()` to accept an initial step too. This can be a vector if particles start at different initial steps, in which case all particles are run up to the latest step (#45)

# dust 0.1.1

* Allow `$set_state()` to accept a matrix in order to start particles with different starting values (#43)
