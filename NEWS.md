# dust 0.8.3

* Reduce host and memcpy usage in device reorder by computing scatter index within the kernel (#198)

# dust 0.8.2

* Don't rewrite files with identical content during generation; this avoids recompilation of code across sessions when the argument `workdir` is used with `dust::dust` (#195)

# dust 0.8.0

* Add GPU support (#73)

# dust 0.7.15

* Add new design vignette - see `vignette("design")` (#161)

# dust 0.7.13

* Improved handling of multi-parameter models, allowing parameter sets to be structured and handling of the special case of one particle per parameter set, outlined in `vignette("multi")` (#169)

# dust 0.7.9

* Deprecate `dust::dust_simulate` and `dust::dust_iterate` which are replaced with a new method `$simulate` on the object which retains state and makes this more powerful (#100, #119, #121)

# dust 0.7.8

* Allow `set_pars` method to be used with multiparameter dust objects (#125)
* Enforce rule that once created a dust object may not change state size (i.e., the number of particles, state elements and number of parameter sets may not change). This was already assumed by mcstate

# dust 0.7.5

* Beginnings of particle filter support, with a new `$filter()` method (only works for models with a compiled "compare" method) (#155)

# dust 0.7.4

* More complete handling of `$compare_data()` with multiple parameters, with the `dust::dust_data` function now expanded to support this (#152)

# dust 0.7.3

* Added new method `$resample()` which implements the resampling algorithm used by the mcstate particle filter (#137)
* The `$compare_data()` method is better behaved with multi-parameter dust objects, returning a matrix (#136)

# dust 0.7.2

* Added new methods `$n_particles()` and `$n_state()` to every dust model which can be used to query the size of the state (#149)

# dust 0.7.0

* Dust models must now specify two internal types, `internal_t` and `shared_t`. The latter is a pointer to constant data shared across all particles within a parameter set (#143)

# dust 0.6.2

* Support for some basic density functions within the header `<dust/densities.hpp>`. Supported distributions include `dbinom`, `dnbinom`, `dbetabinom` and `dpois`. These are included for use within comparison functions (#134)

# dust 0.6.1

* Compiled "comparison" functions are supported, designed to compute likelihoods for mcstate; this interface will be expanded and documented in a future release (#129)

# dust 0.6.0

* The `data` argument (and along with it things like `set_data`) have moved to become `pars` as that is how we're using it, and to make space for a future `data` element (#130)

# dust 0.5.10

* Add support for `rexp` from dust model, just using inversion for now (#127)

# dust 0.5.9

* Start of support for running dust objects with multiple data/parameter sets at once (#92)

# dust 0.5.7

* New method `set_n_threads()` for changing the number of OpenMP threads after initialisation (#122)

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
