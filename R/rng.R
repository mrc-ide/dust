dust_rng <- R6::R6Class(
  "dust_rng",

  private = list(
    ptr = NULL,
    n_generators = NULL
  ),

  public = list(
    initialize = function(seed, n_generators) {
      private$n_generators <- n_generators
      private$ptr <- dust_rng_alloc(seed, n_generators)
    },

    size = function() {
      private$n_generators
    },

    unif_rand = function(n) {
      dust_rng_unif_rand(private$ptr, n)
    },

    norm_rand = function(n) {
      dust_rng_norm_rand(private$ptr, n)
    },

    runif = function(n, min, max) {
      dust_rng_runif(private$ptr, n, recycle(min, n), recycle(max, n))
    },

    rnorm = function(n, mean, sd) {
      dust_rng_rnorm(private$ptr, n, recycle(mean, n), recycle(sd, n))
    },

    rbinom = function(n, size, prob) {
      dust_rng_rbinom(private$ptr, n, recycle(size, n), recycle(prob, n))
    },

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
