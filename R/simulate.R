dust_simulate <- function(model, steps,
                          data = NULL, state = NULL, index = NULL,
                          n_threads = 1L, seed = 1L) {
  if (inherits(model, "dust")) {
    ## Fall back on index and state in the model
    index <- index %||% model$index()
    state <- state %||% model$state()
    data <- data %||% rep(list(model$data()), ncol(state))
    model$simulate(steps, data, state, index, n_threads, seed)
  } else if (inherits(model, "R6ClassGenerator") &&
             identical(model$classname, "dust")) {
    model$public_methods$simulate(steps, data, state, index, n_threads, seed)
  } else {
    stop("Expected a model object or generator for 'model'")
  }
}
