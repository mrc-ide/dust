##' @title Dust Random Number Generator
##'
##' @description Create an object that can be used to generate random
##'   numbers with the same RNG as dust uses internally.  This is
##'   primarily meant for debugging and testing.
##'
##' @export
##' @examples
##' rng <- dust_rng$new(42)
##'
##' # Shorthand for Uniform(0, 1)
##' rng$unif_rand(5)
##'
##' # Shorthand for Normal(0, 1)
##' rng$norm_rand(5)
##'
##' # Uniform random numbers between min and max
##' rng$runif(5, -2, 6)
##'
##' # Normally distributed random numbers with mean and sd
##' rng$rnorm(5, 4, 2)
##'
##' # Binomially distributed random numbers with size and prob
##' rng$rbinom(5, 10L, 0.3)
##'
##' # Poisson distributed random numbers with mean lambda
##' rng$rpois(5, 2)
dust_rng <- R6::R6Class(
  "dust_rng",
  cloneable = FALSE,

  private = list(
    ptr = NULL,
    n_generators = NULL
  ),

  public = list(
    ##' @description Create a `dust_rng` object
    ##'
    ##' @param seed The seed, as an integer
    ##'
    ##' @param n_generators The number of generators to use. While this
    ##'   function never runs in parallel, this is used to create a set of
    ##'   interleaved independent generators as dust would use in a model.
    initialize = function(seed, n_generators = 1L) {
      private$ptr <- dust_rng_alloc(seed, n_generators)
    },

    ##' @description Number of generators available
    size = function() {
      dust_rng_size(private$ptr)
    },

    ##' @description The jump function for the generator, equivalent to
    ##' 2^128 numbers drawn from the generator.
    jump = function() {
      dust_rng_jump(private$ptr)
      invisible(self)
    },

    ##' @description The `long_jump` function for the generator, equivalent
    ##' to 2^192 numbers drawn from the generator.
    long_jump = function() {
      dust_rng_long_jump(private$ptr)
      invisible(self)
    },

    ##' Generate `n` numbers from a standard uniform distribution
    ##'
    ##' @param n Number of samples to draw
    unif_rand = function(n) {
      dust_rng_unif_rand(private$ptr, n)
    },

    ##' Generate `n` numbers from a standard normal distribution
    ##'
    ##' @param n Number of samples to draw
    norm_rand = function(n) {
      dust_rng_norm_rand(private$ptr, n)
    },

    ##' Generate `n` numbers from a uniform distribution
    ##'
    ##' @param n Number of samples to draw
    ##'
    ##' @param min The minimum of the distribution (length 1 or n)
    ##'
    ##' @param max The maximum of the distribution (length 1 or n)
    runif = function(n, min, max) {
      dust_rng_runif(private$ptr, n, recycle(min, n), recycle(max, n))
    },

    ##' Generate `n` numbers from a normal distribution
    ##'
    ##' @param n Number of samples to draw
    ##'
    ##' @param mean The mean of the distribution (length 1 or n)
    ##'
    ##' @param sd The standard deviation of the distribution (length 1 or n)
    rnorm = function(n, mean, sd) {
      dust_rng_rnorm(private$ptr, n, recycle(mean, n), recycle(sd, n))
    },

    ##' Generate `n` numbers from a binomial distribution
    ##'
    ##' @param n Number of samples to draw
    ##'
    ##' @param size The number of trials (zero or more, length 1 or n)
    ##'
    ##' @param prob The probability of success on each trial
    ##'   (between 0 and 1, length 1 or n)
    rbinom = function(n, size, prob) {
      dust_rng_rbinom(private$ptr, n, recycle(size, n), recycle(prob, n))
    },

    ##' Generate `n` numbers from a Poisson distribution
    ##'
    ##' @param n Number of samples to draw
    ##'
    ##' @param lambda The mean (zero or more, length 1 or n)
    rpois = function(n, lambda) {
      dust_rng_rpois(private$ptr, n, recycle(lambda, n))
    }
  ))


recycle <- function(x, n, name = deparse(substitute(x))) {
  if (length(x) == n) {
    x
  } else if (length(x) == 1L) {
    rep(x, n)
  } else {
    stop(sprintf("Invalid length for '%s', expected 1 or %d", name, n))
  }
}
