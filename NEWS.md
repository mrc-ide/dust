# dust 0.13.12

* New method `$time_type()` to determine of a model uses discrete time (`discrete`) or continuous time (`continuous`).

# dust 0.13.10

* Avoid integer overflow in `binomial()`; previously `n` of greater than `2^31 - 1` (`.Machine$integer.max`) would overflow and error (mrc-4046, reported by Jørgen Eriksson Midtbø).

# dust 0.13.8

* Support for solving ODE models (based on a prototype implementation in [mode](https://mrc-ide.github.io/mode); full documentation forthcoming.

# dust 0.12.5

* Allow control over the C++ specification using the argument `cpp_std`, and including other packages' headers using the argument `linking_to` in `dust::dust()`

# dust 0.12.1

* The `update_state()` method gains an `index` argument for setting just some variables in an update.

# dust 0.12.0

* Breaking change, with `step` (and so on with `step_end`) changing to `time` everywhere, in order to smooth the inclusion of continuous time models. This affects quite a few methods:
  * `step()` and `set_step()` become `time()` and `set_time()`
  * `update_state()`'s argument changes from `step` to `time`

# dust 0.11.24

* Add `min_log_likelihood` support to `$filter()` (#361)

# dust 0.11.23

* Allow use of `$filter()` with deterministic models (that support it)
* Allow partial running of the filter from the current step up to some step part way through the data (#177)

# dust 0.11.22

* Allow model names with underscores (#358, see also [odin.dust#98](https://github.com/mrc-ide/odin.dust/issues/98))

# dust 0.11.21

* Remove the functionality that allowed a vector of start times to be provided, as we were never using it and it simplifes things to remove it (#310)

# dust 0.11.17

* Remove `dust::dust_rng_state_long_jump` now that we have proper distributed RNG state support (#343)

# dust 0.11.15

* New support for setting up distributed parallel random seeds (`dust::dust_rng_distributed_state` and `dust::dust_rng_distributed_pointer`), and documentation to guide their use (`vignette("rng_distributed")`) (#297)

# dust 0.11.11

* Removed methods `$set_state`, `$reset` and `$set_pars` which were deprecated in favour of `$update_state` in 0.9.21 (#273)

# dust 0.11.10

* The names of different generator internal states has been simplified to lose the trailing state, e.g. `xoshiro256star_state` becomes `xoshiro256star` (#332)

# dust 0.11.8

* Improved the interface for using dust's random number support from other packages (#329)

# dust 0.11.7

* New polar algorithm for normally distributed random numbers; faster than Box-Muller but slower than Ziggurat
* Tweaked algorithm for Ziggurat so now ~10% faster by avoiding drawing a second random number when working with 64 bit integers
* Added a new vignette describing the internals of the normal sampling algorithms (#325)

# dust 0.11.6

* Added support for drawing normally distributed random numbers using the ziggurat method. Currently this is only efficient on a CPU, though supported on a GPU (#308)

# dust 0.11.5

* Major header reorganisation, the `dust::cuda` namespace is now `dust::gpu`, most user-facing uses of `cuda` and `device` now replaced with `gpu` (#298, #317)
* New (static) method `real_size` on every dust class for accessing the size of `real_type`, in bits (#301)

# dust 0.11.1

* Support for the multinomial distribution. This differs from most other random number support because it returns a *vector* of values not a single number (#307)

# dust 0.11.0

* Rationalised the GPU interface, now once created models can only be used on either the GPU or CPU which simplifies the internal bookkeeping (#292, #302)

# dust 0.10.0

* Improved and generalised RNG interface, with more algorithms and more control
  - Expand set of included generators to 12 with different storage types, period and precision (#281)
  - Faster random number generation with single precision, using `xoshiro128plus` (#282)
  - Slightly faster real number generation for all generators by avoiding division (#280)
  - The `dust_rng` object has been refactored with methods changing names, a new behaviour when using multiple generators, and parallelisation (#279)
* Type names have been standardised and we now avoid `_t` in favour of `_type` in line with the POSIX standard; this impacts all existing dust-using code (#278)
* Density function names have changed from `dust::dbinom` to `dust::density::binomial` (and so on, #291)

# dust 0.9.21

* Deprecate the previous state update methods (`$reset()`, `$set_pars()` and `$set_state()`) in favour of a single method that can update any or all of parameters, model state and time, `$update_state()` (#180)
* Model determinism is now fixed at creation, rather than being settable via `run` and `simulate`, with `deterministic` now an argument to the constructor (#270)

# dust 0.9.20

* Change to the `dust::densities::dnbinom()` to offer both of the same
parameterisations as R's `dnbinom`, explicitly as `dust::densities::dnbinom_mu()` and `dust::densities::dnbinom_prob()` (#171)

# dust 0.9.18

* New function `dust::dust_generate` for creating a mini-package from a dust model for inspection or later loading (#204)
* New option to `dust::dust` to skip the model cache, which may be useful when compiling with (say) different GPU options (#248)

# dust 0.9.14

* Add two new vignettes covering model/data comparison and use on GPUs; see `vignette("data")` and `vignette("cuda")` (#183, #229)

# dust 0.9.10

* Finer control over GPU settings, with the block size of `run()` now (optionally) exposed
* On the GPU integers are kept in shared memory even where reals will no longer fit (#245)

# dust 0.9.8

* Fix infinite loop with rbinom using floats

# dust 0.9.7

* Synchronise possible divergences in the density functions (CUDA only) (#243)

# dust 0.9.6

* Fix a possible issue with dnbinom in float mode with a small mean (#240)

# dust 0.9.5

* Fix a bug when running the systematic resample in the particle filter
in float mode (#238)

# dust 0.9.4

* Fix a bug when running the CUDA version of the particle filter without
history/trajectories.

# dust 0.9.3

* Change `real_t` at compilation, and return information about the size of `real_t` from model objects (#233)

# dust 0.9.2

* Add a CUDA version of the `simulate` method.

# dust 0.9.1

* Move history and snapshot saving out of VRAM, and make it asynchronous.

# dust 0.9.0

* Added CUDA version of the particle filter, run with `model$filter(device = TRUE)` (#224)

# dust 0.8.15

* Removed functions `dust::dust_simulate` and `dust::dust_iterate`, which were deprecated in 0.7.9 (#215)

# dust 0.8.7

* Invalid inputs in `rbinom` are converted into exceptions which are safely thrown even from parallel code (#190)

# dust 0.8.5

* The `filter` method can save snapshots at points along a run (#176)

# dust 0.8.4

* Reduce host and memcpy usage in device reorder by computing scatter index within the kernel (#198)

# dust 0.8.3

* Fix issue with `rnorm()` running on a GPU (device code).
* Fix issue with unaligned shared copy in CUDA code.

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
