##' @title Dust Random Number Generator
##'
##' @description Create an object that can be used to generate random
##'   numbers with the same RNG as dust uses internally.  This is
##'   primarily meant for debugging and testing.
##'
##' @return A `dust_rng` object, which can be used to drawn random
##'   numbers from dust's distributions.
##'
##' @export
##' @examples
##' rng <- dust::dust_rng$new(42)
##'
##' # Shorthand for Uniform(0, 1)
##' rng$unif_rand(5)
##'
##' # Uniform random numbers between min and max
##' rng$uniform(5, -2, 6)
##'
##' # Normally distributed random numbers with mean and sd
##' rng$normal(5, 4, 2)
##'
##' # Binomially distributed random numbers with size and prob
##' rng$binomial(5, 10, 0.3)
##'
##' # Poisson distributed random numbers with mean lambda
##' rng$poisson(5, 2)
##'
##' # Exponentially distributed random numbers with rate
##' rng$exponential(5, 2)
dust_rng <- R6::R6Class(
  "dust_rng",
  cloneable = FALSE,

  private = list(
    ptr = NULL,
    n_generators = NULL,
    float = NULL
  ),

  public = list(
    ##' @field info Information about the generator (read-only)
    info = NULL,

    ##' @description Create a `dust_rng` object
    ##'
    ##' @param seed The seed, as an integer or as a raw vector.
    ##'
    ##' @param n_generators The number of generators to use. While this
    ##'   function never runs in parallel, this is used to create a set of
    ##'   interleaved independent generators as dust would use in a model.
    ##'
    ##' @param real_type The type of floating point number to use. Currently
    ##'   only `float` and `double` are supported (with `double` being
    ##'   the default). This will have no (or negligible) impact on speed,
    ##'   but exists to test the low-precision generators.
    ##'
    ##' @param deterministic Logical, indicating if we should use
    ##'   "deterministic" mode where distributions return their
    ##'   expectations and the state is never changed.
    initialize = function(seed, n_generators = 1L, real_type = "double",
                          deterministic = FALSE) {
      if (!(real_type %in% c("double", "float"))) {
        stop("Invalid value for 'real_type': must be 'double' or 'float'")
      }
      private$float <- real_type == "float"
      private$ptr <- dust_rng_alloc(seed, n_generators, deterministic,
                                    private$float)
      private$n_generators <- n_generators

      if (real_type == "float") {
        size_int_bits <- 32L
        name <- "xoshiro128starstar"
      } else {
        size_int_bits <- 64L
        name <- "xoshiro128starstar"
      }
      size_int_bits <- if (real_type == "float") 32L else 64L
      self$info <- list(
        real_type = real_type,
        int_type = sprintf("uint%s_t", size_int_bits),
        name = name,
        deterministic = deterministic,
        ## Size, in bits, of the underlying integer
        size_int_bits = size_int_bits,
        ## Number of integers used for state
        size_state_ints = 4L,
        ## Total size in bytes of the state
        size_state_bytes = 4L * size_int_bits / 8L)
      lockBinding("info", self)
    },

    ##' @description Number of generators available
    size = function() {
      private$n_generators
    },

    ##' @description The jump function for the generator, equivalent to
    ##' 2^128 numbers drawn from the generator.
    jump = function() {
      dust_rng_jump(private$ptr, private$float)
      invisible(self)
    },

    ##' @description The `long_jump` function for the generator, equivalent
    ##' to 2^192 numbers drawn from the generator.
    long_jump = function() {
      dust_rng_long_jump(private$ptr, private$float)
      invisible(self)
    },

    ##' Generate `n` numbers from a standard uniform distribution
    ##'
    ##' @param n Number of samples to draw
    unif_rand = function(n) {
      dust_rng_random_real(private$ptr, n, private$float)
    },

    ##' Generate `n` numbers from a uniform distribution
    ##'
    ##' @param n Number of samples to draw
    ##'
    ##' @param min The minimum of the distribution (length 1 or n)
    ##'
    ##' @param max The maximum of the distribution (length 1 or n)
    uniform = function(n, min, max) {
      dust_rng_uniform(private$ptr, n, recycle(min, n), recycle(max, n),
                       private$float)
    },

    ##' Generate `n` numbers from a normal distribution
    ##'
    ##' @param n Number of samples to draw
    ##'
    ##' @param mean The mean of the distribution (length 1 or n)
    ##'
    ##' @param sd The standard deviation of the distribution (length 1 or n)
    normal = function(n, mean, sd) {
      dust_rng_normal(private$ptr, n, recycle(mean, n), recycle(sd, n),
                      private$float)
    },

    ##' Generate `n` numbers from a binomial distribution
    ##'
    ##' @param n Number of samples to draw
    ##'
    ##' @param size The number of trials (zero or more, length 1 or n)
    ##'
    ##' @param prob The probability of success on each trial
    ##'   (between 0 and 1, length 1 or n)
    binomial = function(n, size, prob) {
      dust_rng_binomial(private$ptr, n, recycle(size, n), recycle(prob, n),
                        private$float)
    },

    ##' Generate `n` numbers from a Poisson distribution
    ##'
    ##' @param n Number of samples to draw
    ##'
    ##' @param lambda The mean (zero or more, length 1 or n)
    poisson = function(n, lambda) {
      dust_rng_poisson(private$ptr, n, recycle(lambda, n),
                       private$float)
    },

    ##' Generate `n` numbers from a exponential distribution
    ##'
    ##' @param n Number of samples to draw
    ##'
    ##' @param rate The rate of the exponential
    exponential = function(n, rate) {
      dust_rng_exponential(private$ptr, n, recycle(rate, n), private$float)
    },

    ##' @description
    ##' Returns the state of the random number generator. This returns a
    ##' raw vector of length 32 * n_generators. It is primarily intended for
    ##' debugging as one cannot (yet) initialise a dust_rng object with this
    ##' state.
    state = function() {
      dust_rng_state(private$ptr, private$float)
    }
  ))


##' Advance a saved random number state by performing a "long jump" on
##' it. If you have serialised the state using the `$rng_state()`
##' method of a [`dust`] object but want create a new seed that is
##' uncorrelated.  If seed is extracted with `$rng_seed()` is to be
##' reused multiple times, or if it will be used *and* the source
##' object will also be used, then the state needs jumping to prevent
##' generating the same sequence of random numbers.
##'
##' @title Advance a dust random number state
##'
##' @param state A raw vector representing `dust` random number
##'   generator; see [`dust_rng`].
##'
##' @param times An integer indicating the number of times the
##'   `long_jump` should be performed. The default is one, but values
##'   larger than one will repeatedly advance the state.
##'
##' @return A raw vector of random number state, suitable to set into
##'   a `dust` or `dust_rng` object, or for use as a seed.
##'
##' @export
##' @examples
##' # Create a new RNG object
##' rng <- dust::dust_rng$new(1)
##'
##' # Serialise the state as a raw vector
##' state <- rng$state()
##'
##' # We can advance this state
##' dust::dust_rng_state_long_jump(state)
##'
##' # Which gives the same result as long_jump on the original generator
##' rng$long_jump()$state()
##' rng$long_jump()$state()
##'
##' # Multiple jumps can be taken by using the "times" argument
##' dust::dust_rng_state_long_jump(state, 2)
dust_rng_state_long_jump <- function(state, times = 1L) {
  assert_is(state, "raw")
  rng <- dust_rng$new(state)
  for (i in seq_len(times)) {
    rng$long_jump()
  }
  rng$state()
}


recycle <- function(x, n, name = deparse(substitute(x))) {
  if (length(x) == n) {
    x
  } else if (length(x) == 1L) {
    rep_len(x, n)
  } else {
    stop(sprintf("Invalid length for '%s', expected 1 or %d", name, n))
  }
}
