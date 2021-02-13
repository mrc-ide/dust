##' Simulate a dust model.  This function is deprecated since v0.7.8
##' and will be removed in a future version of dust. Please use the
##' `$simulate()` method directly on a dust object - see below for
##' details to update your code.
##'
##' This function was designed to support the special case of
##' simulating a set of trajectories across a list of parameters and
##' initial state. Now that [dust] models support multiple parameters
##' natively this is deprecated.
##'
##' To migrate to use the method, initialise your model with the list
##' of parameters, like:
##'
##' ```
##' mod <- model$new(pars, steps[[1]], 1L)
##' mod$set_state(state)
##' mod$set_index(index) # if using
##' ```
##'
##' In contrast to this function the 3rd argument there can be used to
##' simulate multiple trajectories *per* parameter set. You can set
##' `n_threads` and `seed` as in the constructor as normal.
##'
##' Now you can run the model with
##'
##' ```
##' y <- mod$simulate(steps)
##' ```
##'
##' which will return a 4d matrix (in this case with 1 trajectory per
##' parameter set.
##'
##' The advantage of this approach over the previous `dust_simulate`
##' approach is that now you can inspect your model to get state,
##' continue it etc without having to worry about the rng state; it
##' should be much more flexible.
##'
##' @title Simulate from a dust model generator
##'
##' @param model A [dust] model generator object
##'
##' @param steps The vector of steps used to simulate over. The first
##'   step in this vector is the starting step (corresponding to the
##'   initial `state` of the model) and subsequent values are steps
##'   where state should be returned.
##'
##' @param pars An unnamed list of model initialisation pars (see
##'   [dust]).  It must have the same length as the number of columns
##'   in `state`.
##'
##' @param state A matrix of initial states. It must have the number
##'   of rows corresponding to the number of initial state values your
##'   model requires, and the number of columns corresponding to the
##'   number of independent simulations to perform (i.e.,
##'   `length(pars)`).
##'
##' @param index An optional index, indicating the indicies of the
##'   state vectors that you want output recorded for.
##'
##' @param n_threads Number of OMP threads to use, if `dust` and your
##'   model were compiled with OMP support.  The number of simulations
##'   (`length(pars)`) should be a multiple of `n_threads` (e.g., if
##'   you use 8 threads, then you should have 8, 16, 24, etc
##'   particles). However, this is not compulsary.
##'
##' @param seed The seed to use for the random number generator. Can
##'   be a positive integer, `NULL` (initialise with R's random number
##'   generator) or a `raw` vector of a length that is a multiple of
##'   32 to directly initialise the generator (e.g., from the
##'   [`dust`] object's `$rng_state()` method).
##'
##' @param return_state Logical, indicating if the final state should
##'   be returned. If `TRUE`, then an attribute `state` with the same
##'   dimensions as the input `state` will be added to the array,
##'   along with an attribute `rng_state` with the internal state of
##'   the random number generator.
##'
##' @export
##' @examples
##' # Use the "random walk" example
##' model <- dust::dust_example("walk")
##'
##' # Start with 40 parameter sets; for this model each is list with
##' # an element 'sd'
##' pars <- replicate(40, list(sd = runif(1)), simplify = FALSE)
##'
##' # We also need a matrix of initial states
##' y0 <- matrix(rnorm(40), 1, 40)
##'
##' # Run from steps 0..50
##' steps <- 0:50
##'
##' # The simulated output:
##' res <- dust::dust_simulate(model, steps, pars, y0)
##'
##' # The result of the simulation, plotted over time
##' matplot(steps, t(drop(res)), type = "l", col = "#00000055", lty = 1)
dust_simulate <- function(model, steps, pars, state, index = NULL,
                          n_threads = 1L, seed = NULL, return_state = FALSE) {
  if (inherits(model, "dust")) {
    stop("dust_simulate no longer valid for dust models")
  } else if (inherits(model, "R6ClassGenerator") &&
             identical(model$classname, "dust")) {
    .Deprecated("$simulate() method directly (see ?dust_simulate)")
    simulate <- model$private_methods$simulate
  } else {
    stop("Expected a dust generator for 'model'")
  }
  if (!is.matrix(state)) {
    stop("Expected 'state' to be a matrix")
  }
  if (is.list(pars) && !is.null(names(pars))) {
    stop("Expected 'pars' to be an unnamed list")
  }
  if (length(pars) != ncol(state)) {
    stop(sprintf("Expected 'state' to be a matrix with %d columns",
                 length(pars)))
  }

  mod <- model$new(pars, steps[[1]], 1L,
                   n_threads = n_threads, seed = seed, pars_multi = TRUE)
  dim(state) <- c(nrow(state), 1, ncol(state))
  mod$set_state(state)
  if (!is.null(index)) {
    mod$set_index(index)
  }
  ret <- mod$simulate2(steps)
  dim(ret) <- dim(ret)[-2L]
  if (return_state) {
    y <- mod$state()
    dim(y) <- dim(y)[-2L]
    attr(ret, "state") <- y
    attr(ret, "rng_state") <- mod$rng_state()
  }
  ret
}
