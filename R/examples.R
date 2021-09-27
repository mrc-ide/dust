##' Access dust's built-in examples. These are compiled into the
##' package so that examples and tests can be run more quickly without
##' having to compile code directly via [dust()].  These examples are
##' all "toy" examples, being small and fast to run.
##'
##' * `sir`: a basic SIR (Susceptible, Infected, Resistant)
##'   epidemiological model.  Draws from the binomial distribution to
##'   update the population between each step.
##'
##' * `sirs`: an SIRS model, the SIR model with an added R->S transition.
##'  This has a non-zero steady state, so can be run indefinitely for testing.
##'
##' * `volatility`: A volatility model that might be applied to
##'   currency fluctuations etc.
##'
##' * `walk`: A 1D random walk, following a Gaussian distribution each
##'   time step.
##'
##' @title Access dust's built-in examples
##'
##' @param name The name of the example to use. There are four
##'   examples: `sir`, `sirs`, `variable`, `volatility` and `walk`
##'   (see Details).
##'
##' @return A [`dust::dust_generator`] object that can be used to create a
##'   model.  See examples for usage.
##'
##' @export
##' @examples
##'
##' # A SIR (Susceptible, Infected, Resistant) epidemiological model
##' sir <- dust::dust_example("sir")
##' sir
##'
##' # Initialise the model at step 0 with 50 independent trajectories
##' mod <- sir$new(list(), 0, 50)
##'
##' # Run the model for 400 steps, collecting "infected" every 4th step
##' steps <- seq(0, 400, by = 4)
##' mod$set_index(2L)
##' y <- mod$simulate(steps)
##'
##' # A plot of our epidemic
##' matplot(steps, t(drop(y)), type = "l", lty = 1, col = "#00000044",
##'         las = 1, xlab = "Step", ylab = "Number infected")
dust_example <- function(name) {
  ## NOTE: the documentation does not mention 'variable' as it's
  ## primarily for testing the package behaviour as the number of
  ## state variables change: we'll probably swap it out for something
  ## more interesting later?
  switch(name,
         sir = sir,
         sirs = sirs,
         variable = variable,
         volatility = volatility,
         walk = walk,
         stop(sprintf("Unknown example '%s'", name)))
}
