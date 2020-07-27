volatilitygpu <- R6::R6Class(
  "dust",
  cloneable = FALSE,

  private = list(
    data = NULL,
    ptr = NULL
  ),

  public = list(
    ##' @description
    ##' Create a new model
    ##'
    ##' @param data Data to initialise your model with; a `list`
    ##' object, but the required elements will depend on the details of
    ##' your model.
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
    ##' is not compulsary.
    ##'
    ##' @param seed Seed to use for the random number generator
    ##' (positive integer)
    initialize = function(data, step, n_particles, n_threads = 1L,
                          seed = 1L) {
      res <- dust_volatilitygpu_alloc(data, step, n_particles,
                        n_threads, seed)
      private$ptr <- res[[1L]]
      private$data <- res[[2L]]
    },

    ##' @description
    ##' Returns friendly model name
    name = function() {
      "volatilitygpu"
    },

    ##' @description
    ##' Run the model up to a point in time, returning the filtered state
    ##' at that point.
    ##'
    ##' @param step_end Step to run to (if less than or equal to the current
    ##' step(),silently nothing will happen)
    run = function(step_end) {
      dust_volatilitygpu_run(private$ptr, step_end)
    },

    ##' @description
    ##' Set the "index" vector that is used to return a subset of data
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
      dust_volatilitygpu_set_index(private$ptr, index)
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
    ##' larges step given (i.e., `max(step)`)
    set_state = function(state, step = NULL) {
      dust_volatilitygpu_set_state(private$ptr, state, step)
    },

    ##' @description
    ##' Reset the model while preserving the random number stream state
    ##'
    ##' @param data New data for the model (see constructor)
    ##' @param step New initial step for the model (see constructor)
    reset = function(data, step) {
      private$data <- dust_volatilitygpu_reset(private$ptr, data, step)
      invisible()
    },

    ##' @description
    ##' Return full model state
    ##' @param index Optional index to select state using
    state = function(index = NULL) {
      dust_volatilitygpu_state(private$ptr, index)
    },

    ##' @description
    ##' Return current model step
    step = function() {
      dust_volatilitygpu_step(private$ptr)
    },

    ##' @description
    ##' Reorder or resample particles.
    ##' @param index An integer vector, with values between 1 and n_particles,
    ##' indicating the index of the current particles that new particles should
    ##' take.
    reorder = function(index) {
      dust_volatilitygpu_reorder(private$ptr, as.integer(index))
      invisible()
    },

    ##' @description
    ##' Returns information about the data that your model was created with.
    ##' Only returns non-NULL if the model provides a `dust_info` template
    ##' specialisation.
    info = function() {
      private$data
    },

    ##' @description
    ##' Returns the state of the random number generator. This returns a
    ##' raw vector of length 32 * n_particles. It is primarily intended for
    ##' debugging as one cannot (yet) initialise a dust object with this
    ##' state.
    rng_state = function() {
      dust_volatilitygpu_rng_state(private$ptr)
    }
  ))
