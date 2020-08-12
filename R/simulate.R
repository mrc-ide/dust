##' Simulate a dust model.  This is a helper function that is subject
##' to change; see Details below.
##'
##' This function has an interface that we expect to change once
##' multi-parameter "dust" objects are fully supported.  For now, it
##' is designed to be used where we want to simulate a number of
##' trajectories given a vector of model data and a matrix of initial
##' state.  For our immediate use case this is for simulating at the
##' end of an MCMC where want to generate a posterior distribution of
##' trajectories.
##'
##' @section Random number generation:
##'
##' This implementation leaves a number of issues that we will resolve
##'   in future versions. Most pressing is that the RNG is uncoupled
##'   between a `model` object and the simulation; ideally this would
##'   advance the RNG on the `model`, but this needs care in the case
##'   where either more or fewer simulations are carried out than the
##'   initial object has.  This is resolvable with some support for
##'   setting rng state directly from a raw vector and with jumping
##'   forward for the the component (model object or simulation) that
##'   has fewer particles.
##'
##' @title Simulate from a model or generator
##'
##' @param model A [dust] model or generator object
##'
##' @param steps The vector of steps used to simulate over. The first
##'   step in this vector is the starting step (corresponding to the
##'   initial `state` of the model) and subsequent values are steps
##'   where state should be returned.
##'
##' @param data An unnamed list of model initialisation data (see
##'   [dust]).  It must have the same length as the number of columns
##'   in `state`.
##'
##' @param state A matrix of initial states. It must have the number
##'   of rows corresponding to the number of initial state values your
##'   model requires, and the number of columns corresponding to the
##'   number of independent simulations to perform (i.e.,
##'   `length(data)`).
##'
##' @param index An optional index, indicating the indicies of the
##'   state vectors that you want output recorded for.
##'
##' @param n_threads Number of OMP threads to use, if `dust` and your
##'   model were compiled with OMP support.  The number of simulations
##'   (`length(data)`) should be a multiple of `n_threads` (e.g., if
##'   you use 8 threads, then you should have 8, 16, 24, etc
##'   particles). However, this is not compulsary.
##'
##' @param seed The seed to use for the random number generator
##'   (positive integer)
##'
##' @export
dust_simulate <- function(model, steps, data, state, index = NULL,
                          n_threads = 1L, seed = 1L) {
  if (inherits(model, "dust")) {
    simulate <- environment(model$run)$private$simulate
  } else if (inherits(model, "R6ClassGenerator") &&
             identical(model$classname, "dust")) {
    simulate <- model$private_methods$simulate
  } else {
    stop("Expected a model object or generator for 'model'")
  }
  if (!is.matrix(state)) {
    stop("Expected 'state' to be a matrix")
  }
  if (is.list(data) && !is.null(names(data))) {
    stop("Expected 'data' to be an unnamed list")
  }
  simulate(steps, data, state, index, n_threads, seed)
}
