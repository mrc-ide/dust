## DO NOT EDIT THIS FILE!
## Instead, edit the file inst/template/dust.R.template
## and run ./scripts/update_dust_class

## NOTE: R6 classes do not support interfaces and inheritence is not
## really needed here as this does not *do* anything, so consider this
## a hack to allow Roxygen's R6 documentation to work.

## The code and object below have exactly the interface of all generated
## dust objects, and this page acts as a reference for all such methods
## modelled on the ?regex page in base R's documentation
##' @name dust_class
##' @rdname dust_class
##' @title The dust class
##'
##' @description
##'
##' All "dust" dust models are really [R6][R6::R6Class] objects and
##'   expose the same set of "methods".  To create a dust model of
##'   your own, see [dust::dust] and to interact with some built-in
##'   ones see [dust::dust_example()]
dust_class <- R6::R6Class(
  "dust",
  cloneable = FALSE,

  private = list(
    pars_ = NULL,
    pars_multi_ = NULL,
    index_ = NULL,
    info_ = NULL,
    n_threads_ = NULL,
    n_particles_ = NULL,
    n_particles_each_ = NULL,
    shape_ = NULL,
    ptr_ = NULL,
    device_config_ = NULL,
    param_ = NULL
  ),

  public = list(
    ##' @description
    ##' Create a new model. Note that the behaviour of this object
    ##' created by this function will change considerably based on
    ##' whether the `pars_multi` argument is `TRUE`. If not (the
    ##' default) then we create `n_particles` which all share the same
    ##' parameters as specified by the `pars` argument. If `pars_multi`
    ##' is `TRUE` then `pars` must be an unnamed list, and each element
    ##' of it represents a different set of parameters. We will
    ##' create `length(pars)` *sets* of `n_particles` particles which
    ##' will be simulated together. These particles must have the same
    ##' dimension - that is, they must correspond to model state that
    ##' is the same size.
    ##'
    ##' @param pars Data to initialise your model with; a `list`
    ##' object, but the required elements will depend on the details of
    ##' your model. If `pars_multi` is `TRUE`, then this must be an
    ##' *unnamed* list of `pars` objects (see Details).
    ##'
    ##' @param step Initial step - must be nonnegative
    ##'
    ##' @param n_particles Number of particles to create - must be at
    ##' least 1
    ##'
    ##' @param n_threads Number of OMP threads to use, if `dust` and
    ##' your model were compiled with OMP support (details to come).
    ##' `n_particles` should be a multiple of `n_threads` (e.g., if you use 8
    ##' threads, then you should have 8, 16, 24, etc particles). However, this
    ##' is not compulsory.
    ##'
    ##' @param seed The seed to use for the random number generator. Can
    ##' be a positive integer, `NULL` (initialise with R's random number
    ##' generator) or a `raw` vector of a length that is a multiple of
    ##' 32 to directly initialise the generator (e..g., from the
    ##' [`dust`] object's `$rng_state()` method).
    ##'
    ##' @param pars_multi Logical, indicating if `pars` should be
    ##' interpreted as a set of different initialisations, and that we
    ##' should prepare `n_particles * length(pars)` particles for
    ##' simulation. This has an effect on many of the other methods of
    ##' the object.
    ##'
    ##' @param device_config Device configuration, typically an integer
    ##' indicating the device to use, where the model has GPU support.
    ##' If not given, then the default value of `NULL` will fall back on the
    ##' first found device if any are available. An error is thrown if the
    ##' device id given is larger than those reported to be available (note
    ##' that CUDA numbers devices from 0, so that '0' is the first device,
    ##' and so on). Negative values disable the use of a device. See the
    ##' method `$device_info()` for available device ids; this can be called
    ##' before object creation as `dust_class$public_methods$device_info()`.
    ##' For additional control, provide a list with elements `device_id`
    ##' and `run_block_size`. Further options (and validation) of this
    ##' list will be added in a future version!
    initialize = function(pars, step, n_particles, n_threads = 1L,
                          seed = NULL, pars_multi = FALSE,
                          device_config = NULL) {
    },

    ##' @description
    ##' Returns friendly model name
    name = function() {
    },

    ##' @description
    ##' Returns parameter information, if provided by the model. This
    ##' describes the contents of pars passed to the constructor or to
    ##' `reset` as the `pars` argument, and the details depend on the model.
    param = function() {
    },

    ##' @description
    ##' Run the model up to a point in time, returning the filtered state
    ##' at that point.
    ##'
    ##' @param step_end Step to run to (if less than or equal to the current
    ##' step(), silently nothing will happen)
    ##'
    ##' @param device **Experimental!**: This argument may allow running on
    ##' a GPU once support is finished, if the model supports it, and if
    ##' the model is compiled appropriately (and assuming you have a
    ##' suitable GPU). At present it exists for testing and will run
    ##' slower than running with `device = TRUE`. The interpretation of
    ##' this argument will likely change to allow selecting the GPU on
    ##' systems with more than one. In short, please leave this argument
    ##' alone unless you're developing dust.
    run = function(step_end, device = FALSE) {
    },

    ##' @description
    ##' Iterate all particles forward in time over a series of steps,
    ##' collecting output as they go. This is a helper around `$run()`
    ##' where you want to run to a series of points in time and save
    ##' output. The returned object will be filtered by your active index,
    ##' so that it has shape (`n_state` x `n_particles` x `length(step_end)`)
    ##' for single-parameter objects, and (`n_state` x `n_particles` x
    ##' `n_pars` x `length(step_end)`) for multiparameter objects. Note that
    ##' this method is very similar to `$run()` except that the rank of
    ##' the returned array is one less. For a scalar `step_end` you would
    ##' ordinarily want to use `$run()` but the resulting numbers would
    ##' be identical.
    ##'
    ##' @param step_end A vector of time points that the simulation should
    ##'   report output at. This the first time must be at least the same
    ##'   as the current time, and every subsequent time must be equal or
    ##'   greater than those before it (ties are allowed though probably
    ##'   not wanted).
    ##'
    ##' @param device **Experimental!**: This argument may allow running on
    ##' a GPU once support is finished, if the model supports it, and if
    ##' the model is compiled appropriately (and assuming you have a
    ##' suitable GPU). At present it exists for testing and will run
    ##' slower than running with `device = TRUE`. The interpretation of
    ##' this argument will likely change to allow selecting the GPU on
    ##' systems with more than one. In short, please leave this argument
    ##' alone unless you're developing dust.
    simulate = function(step_end, device = FALSE) {
    },

    ##' @description
    ##' Set the "index" vector that is used to return a subset of pars
    ##' after using `run()`. If this is not used then `run()` returns
    ##' all elements in your state vector, which may be excessive and slower
    ##' than necessary. This method must be called after any
    ##' call to `reset()` as `reset()` may change the size of the state
    ##' and that will invalidate the index.
    ##'
    ##' @param index The index vector - must be an integer vector with
    ##' elements between 1 and the length of the state (this will be
    ##' validated, and an error thrown if an invalid index is given).
    set_index = function(index) {
    },

    ##' @description
    ##' Returns the `index` as set by `$set_index`
    index = function() {
    },

    ##' @description
    ##' Returns the number of threads that the model was constructed with
    n_threads = function() {
    },

    ##' @description
    ##' Returns the length of the per-particle state
    n_state = function() {
    },

    ##' @description
    ##' Returns the number of particles
    n_particles = function() {
    },

    ##' @description
    ##' Returns the number of particles per parameter set
    n_particles_each = function() {
    },

    ##' @description
    ##' Returns the shape of the particles
    shape = function() {
    },

    ##' @description
    ##' Set the "state" vector for all particles, overriding whatever your
    ##' models `initial()` method provides.
    ##'
    ##' @param state The state vector - can be either a numeric vector with the
    ##' same length as the model's current state (in which case the same
    ##' state is applied to all particles), or a numeric matrix with as
    ##' many rows as your model's state and as many columns as you have
    ##' particles (in which case you can set a number of different starting
    ##' states at once).
    ##'
    ##' @param step If not `NULL`, then this sets the initial step. If this
    ##' is a vector (with the same length as the number of particles), then
    ##' particles are started from different initial steps and run up to the
    ##' largest step given (i.e., `max(step)`)
    set_state = function(state, step = NULL) {
    },

    ##' @description
    ##' Reset the model while preserving the random number stream state
    ##'
    ##' @param pars New pars for the model (see constructor)
    ##' @param step New initial step for the model (see constructor)
    reset = function(pars, step) {
    },

    ##' @description
    ##' Set the 'pars' element in a dust object while holding model state,
    ##' index, etc constant. In contrast to `$reset`, the old state must
    ##' be compatible with the new one (e.g., don't change model size), and
    ##' the index will remain valid.
    ##'
    ##' @param pars New pars for the model (see constructor)
    set_pars = function(pars) {
    },

    ##' @description
    ##' Return full model state
    ##' @param index Optional index to select state using
    state = function(index = NULL) {
    },

    ##' @description
    ##' Return current model step
    step = function() {
    },

    ##' @description
    ##' Reorder particles.
    ##' @param index An integer vector, with values between 1 and n_particles,
    ##' indicating the index of the current particles that new particles should
    ##' take.
    reorder = function(index) {
    },

    ##' @description
    ##' Resample particles according to some weight.
    ##'
    ##' @param weights A numeric vector representing particle weights.
    ##' For a "multi-parameter" dust object this should be be a matrix
    ##' with the number of rows being the number of particles per
    ##' parameter set and the number of columns being the number of
    ##' parameter sets.
    ##' long as all particles or be a matrix.
    resample = function(weights) {
    },

    ##' @description
    ##' Returns information about the pars that your model was created with.
    ##' Only returns non-NULL if the model provides a `dust_info` template
    ##' specialisation.
    info = function() {
    },

    ##' @description
    ##' Returns the `pars` object that your model was constructed with.
    pars = function() {
    },

    ##' @description
    ##' Returns the state of the random number generator. This returns a
    ##' raw vector of length 32 * n_particles. This can be useful for
    ##' debugging or for initialising other dust objects. The arguments
    ##' `first_only` and `last_only` are mutally exclusive. If neither is
    ##' given then all all particles states are returned, being 32 bytes
    ##' per particle. The full returned state or `first_only` are most
    ##' suitable for reseeding a new dust object.
    ##'
    ##' @param first_only Logical, indicating if we should return only the
    ##'   first random number state
    ##'
    ##' @param last_only Logical, indicating if we should return only the
    ##' *last* random number state, which does not belong to a particle.
    rng_state = function(first_only = FALSE, last_only = FALSE) {
    },

    ##' @description Set the random number state for this model. This
    ##' replaces the RNG state that the model is using with a state of
    ##' your choosing, saved out from a different model object. This method
    ##' is designed to support advanced use cases where it is easier to
    ##' manipulate the state of the random number generator than the
    ##' internal state of the dust object.
    ##'
    ##' @param rng_state A random number state, as saved out by the
    ##' `$rng_state()` method. Note that unlike `seed` as passed to the
    ##' constructor, this *must* be a raw vector of the expected length.
    set_rng_state = function(rng_state) {
    },

    ##' @description
    ##' Returns a logical, indicating if this model was compiled with
    ##' "OpenMP" support, in which case it will react to the `n_threads`
    ##' argument passed to the constructor. This method can also be used
    ##' as a static method by running it directly
    ##' as `dust_class$public_methods$has_openmp()`
    has_openmp = function() {
    },

    ##' @description
    ##' Returns a logical, indicating if this model was compiled with
    ##' "CUDA" support, in which case it will react to the `device`
    ##' argument passed to the run method. This method can also be used
    ##' as a static method by running it directly
    ##' as `dust_class$public_methods$has_cuda()`
    has_cuda = function() {
    },

    ##' @description
    ##' Returns the number of distinct pars elements required. This is `0`
    ##' where the object was initialised with `pars_multi = FALSE` and
    ##' an integer otherwise.  For multi-pars dust objects, Where `pars`
    ##' is accepted, you must provide an unnamed list of length `$n_pars()`.
    n_pars = function() {
    },

    ##' @description
    ##' Change the number of threads that the dust object will use. Your
    ##' model must be compiled with "OpenMP" support for this to have an
    ##' effect. Returns (invisibly) the previous value.
    ##'
    ##' @param n_threads The new number of threads to use. You may want to
    ##'   wrap this argument in [dust::dust_openmp_threads()] in order to
    ##'   verify that you can actually use the number of threads
    ##'   requested (based on environment variables and OpenMP support).
    set_n_threads = function(n_threads) {
    },

    ##' @description
    ##' Returns a logical, indicating if this model was compiled with
    ##' "compare" support, in which case the `set_data` and `compare_data`
    ##' methods are available (otherwise these methods will error). This
    ##' method can also be used as a static method by running it directly
    ##' as `dust_class$public_methods$has_compare()`
    has_compare = function() {
    },

    ##' @description
    ##' Set "data" into the model for use with the `$compare_data()` method.
    ##' This is not supported by all models, depending on if they define a
    ##' `data_t` type.  See [dust::dust_data()] for a helper function to
    ##' construct suitable data and a description of the required format. You
    ##' will probably want to use that here, and definitely if using multiple
    ##' parameter sets.
    ##'
    ##' @param data A list of data to set.
    set_data = function(data) {
    },

    ##' @description
    ##' Compare the current model state against the data as set by
    ##' `set_data`. If there is no data set, or no data corresponding to
    ##' the current time then `NULL` is returned. Otherwise a numeric vector
    ##' the same length as the number of particles is returned. If model's
    ##' underlying `compare_data` function is stochastic, then each call to
    ##' this function may be result in a different answer.
    ##'
    ##' @param device **Experimental!**: This argument may allow running on
    ##' a GPU once support is finished, if the model supports it, and if
    ##' the model is compiled appropriately (and assuming you have a
    ##' suitable GPU). At present it exists for testing and will run
    ##' slower than running with `device = FALSE`. The interpretation of
    ##' this argument will likely change to allow selecting the GPU on
    ##' systems with more than one. In short, please leave this argument
    ##' alone unless you're developing dust.
    compare_data = function(device = FALSE) {
    },

    ##' @description
    ##' Run a particle filter. The interface here will change a lot over the
    ##' next few versions. You *must* `$reset()` the filter before using
    ##' this method to get sensible values. We will tinker with this in
    ##' future versions to allow things like partial runs.
    ##'
    ##' @param save_trajectories Logical, indicating if the filtered particle
    ##' trajectories should be saved. If `TRUE` then the `trajectories` element
    ##' will be a multidimensional array (`state x <shape> x time`)
    ##' containing the state values, selected according to the index set
    ##' with `$set_index()`.
    ##'
    ##' @param step_snapshot Optional integer vector indicating steps
    ##' that we should record a snapshot of the full particle filter state.
    ##' If given it must be strictly increasing vector whose elements
    ##' match steps given in the `data` object. The return value with be
    ##' a multidimensional array (`state x <shape> x step_snapshot`)
    ##' containing full state values at the requested steps.
    ##'
    ##' @param device **Experimental!**: This argument may allow running on
    ##' a GPU once support is finished, if the model supports it, and if
    ##' the model is compiled appropriately (and assuming you have a
    ##' suitable GPU). At present it exists for testing and will run
    ##' slower than running with `device = FALSE`. The interpretation of
    ##' this argument will likely change to allow selecting the GPU on
    ##' systems with more than one. In short, please leave this argument
    ##' alone unless you're developing dust.
    filter = function(save_trajectories = FALSE, step_snapshot = NULL,
                      device = FALSE) {
    },

    ##' @description
    ##' **Experimental!** Return information about GPU devices, if the model
    ##' has been compiled with CUDA/GPU support. This can be called as a
    ##' static method by running `dust_class$public_methods$device_info()`.
    ##' If run from a GPU enabled object, it will also have an element
    ##' `config` containing the computed device configuration: the device
    ##' id, shared memory and the block size for the `run` method on the
    ##' device.
    device_info = function() {
    }
  ))
class(dust_class) <- c("dust_generator", class(dust_class))
