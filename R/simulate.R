##' Simulate a dust model over time. This is a wrapper around calling
##' `$run()` and `$state()` repeatedly without doing anything very
##' interesting with the data. It is provided mostly as a
##' quick-and-dirty way of getting started with a model.
##'
##' @title Simulate a dust model
##'
##' @param model A model, compiled with `dust::dust()` and initialised
##'
##' @param steps A vector of steps - the first step must be same as
##'   the model's step (i.e., `$step()`)
##'
##' @param index An optional index to filter the results with.
##'
##' @return A 3d array of model outputs. The first dimension is model
##'   state, the second is particle number and the third is time, so
##'   that `output[i, j, k]` is the `i`th variable, `j`th particle and
##'   `k`th step.
##'
##' @export
##' @examples
##' # Same random walk example as in ?dust
##' filename <- system.file("examples/walk.cpp", package = "dust")
##' model <- dust::dust(filename, quiet = TRUE)
##'
##' # Create a model with 100 particles, starting at step 0
##' obj <- model$new(list(sd = 1), 0, 100)
##'
##' # Steps that we want to report at:
##' steps <- seq(0, 400, by = 4)
##'
##' # Run the simulation:
##' res <- dust::dust_simulate(obj, steps)
##'
##' # Output is 1 x 100 x 100 (state, particle, time)
##' dim(res)
##'
##' # Dropping the first dimension and plotting, with the mean in red
##' # and the expectation in blue:
##' xy <- t(res[1, , , drop = TRUE])
##' matplot(steps, xy, type = "l", lty = 1, col = "#00000033",
##'         xlab = "Step", ylab = "Value")
##' abline(h = 0, lty = 2, col = "blue")
##' lines(steps, rowMeans(xy), col = "red", lwd = 2)
dust_simulate <- function(model, steps, index = NULL) {
  assert_is(model, "dust")
  if (model$step() != steps[[1]]) {
    stop(sprintf("Expected first 'steps' element to be %d", model$step()))
  }
  y <- model$state(index)
  res <- array(NA_real_, dim = c(dim(y), length(steps)))
  for (i in seq_along(steps)) {
    model$run(steps[[i]])
    res[, , i] <- model$state(index)
  }
  res
}
