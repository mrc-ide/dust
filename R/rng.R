##' @title Dust Random Number Generator
##'
##' @description Create an object that can be used to generate random
##'   numbers with the same RNG as dust uses internally.  This is
##'   primarily meant for debugging and testing the underlying C++
##'   rather than a source of random numbers from R.
##'
##' @section Running multiple streams, perhaps in parallel:
##'
##' The underlying random number generators are designed to work in
##'   parallel, and with random access to parameters (see
##'   `vignette("rng")` for more details).  However, this is usually
##'   done within the context of running a model where each particle
##'   sees its own stream of numbers.  We provide some support for
##'   running random number generators in parallel, but any speed
##'   gains from parallelisation are likely to be somewhat eroded by
##'   the overhead of copying around a large number of random numbers.
##'
##' All the random distribution functions support an argument
##'   `n_threads` which controls the number of threads used.  This
##'   argument will *silently* have no effect if your installation
##'   does not support OpenMP (see [dust::dust_openmp_support]).
##'
##' Parallelisation will be performed at the level of the stream,
##'   where we draw `n` numbers from *each* stream for a total of `n *
##'   n_streams` random numbers using `n_threads` threads to do this.
##'   Setting `n_threads` to be higher than `n_streams` will therefore
##'   have no effect. If running on somebody else's system (e.g., an
##'   HPC, CRAN) you must respect the various environment variables
##'   that control the maximum allowable number of threads; consider
##'   using [dust::dust_openmp_threads] to select a safe number.
##'
##' With the exception of `random_real`, each random number
##'   distribution accepts parameters; the interpretations of these
##'   will depend on `n`, `n_streams` and their rank.
##'
##'   * If a scalar then we will use the same parameter value for every draw
##'     from every stream
##'
##'   * If a vector with length `n` then we will draw `n` random
##'     numbers per stream, and every stream will use the same parameter
##'     value for every stream for each draw (but a different,
##'     shared, parameter value for subsequent draws).
##'
##'   * If a matrix is provided with one row and `n_streams`
##'     columns then we use different parameters for each stream, but
##'     the same parameter for each draw.
##'
##'   * If a matrix is provided with `n` rows and `n_streams`
##'     columns then we use a parameter value `[i, j]` for the `i`th
##'     draw on the `j`th stream.
##'
##' The rules are slightly different for the `prob` argument to
##'   `multinomial` as for that `prob` is a vector of values. As such
##'   we shift all dimensions by one:
##'
##'   * If a vector we use same `prob` every draw from every stream
##'     and there are `length(prob)` possible outcomes.
##'
##'   * If a matrix with `n` columns then vary over each draw (the
##'     `i`th draw using vector `prob[, i]` but shared across all
##'     streams. There are `nrow(prob)` possible outcomes.
##'
##'   * If a 3d array is provided with 1 column and `n_streams`
##'     "layers" (the third dimension) then we use then we use different
##'     parameters for each stream, but the same parameter for each
##'     draw.
##'
##'   * If a 3d array is provided with `n` columns and `n_streams`
##'     "layers" then we vary over both draws and streams so that with
##'     use vector `prob[, i, j]` for the `i`th draw on the `j`th
##'     stream.
##'
##' The output will not differ based on the number of threads used,
##'   only on the number of streams.
##'
##' @return A `dust_rng` object, which can be used to drawn random
##'   numbers from dust's distributions.
##'
##' @export
##' @examples
##' rng <- dust::dust_rng$new(42)
##'
##' # Shorthand for Uniform(0, 1)
##' rng$random_real(5)
##'
##' # Shorthand for Normal(0, 1)
##' rng$random_normal(5)
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
##' # Negative binomially distributed random numbers with size and prob
##' rng$nbinomial(5, 10, 0.3)
##'
##' # Hypergeometric distributed random numbers with parameters n1, n2 and k
##' rng$hypergeometric(5, 6, 10, 4)
##'
##' # Gamma distributed random numbers with parameters a and b
##' rng$gamma(5, 0.5, 2)
##'
##' # Poisson distributed random numbers with mean lambda
##' rng$poisson(5, 2)
##'
##' # Exponentially distributed random numbers with rate
##' rng$exponential(5, 2)
##'
##' # Multinomial distributed random numbers with size and vector of
##' # probabiltiies prob
##' rng$multinomial(5, 10, c(0.1, 0.3, 0.5, 0.1))
dust_rng <- R6::R6Class(
  "dust_rng",
  cloneable = FALSE,

  private = list(
    ptr = NULL,
    n_streams = NULL,
    float = NULL
  ),

  public = list(
    ##' @field info Information about the generator (read-only)
    info = NULL,

    ##' @description Create a `dust_rng` object
    ##'
    ##' @param seed The seed, as an integer, a raw vector or `NULL`.
    ##'   If an integer we will create a suitable seed via the "splitmix64"
    ##'   algorithm, if a raw vector it must the correct length (a multiple
    ##'   of either 32 or 16 for `float = FALSE` or `float = TRUE`
    ##'   respectively). If `NULL` then we create a seed using R's random
    ##'   number generator.
    ##'
    ##' @param n_streams The number of streams to use (see Details)
    ##'
    ##' @param real_type The type of floating point number to use. Currently
    ##'   only `float` and `double` are supported (with `double` being
    ##'   the default). This will have no (or negligible) impact on speed,
    ##'   but exists to test the low-precision generators.
    ##'
    ##' @param deterministic Logical, indicating if we should use
    ##'   "deterministic" mode where distributions return their
    ##'   expectations and the state is never changed.
    initialize = function(seed = NULL, n_streams = 1L, real_type = "double",
                          deterministic = FALSE) {
      if (!(real_type %in% c("double", "float"))) {
        stop("Invalid value for 'real_type': must be 'double' or 'float'")
      }
      private$float <- real_type == "float"
      private$ptr <- dust_rng_alloc(seed, n_streams, deterministic,
                                    private$float)
      private$n_streams <- n_streams

      if (real_type == "float") {
        size_int_bits <- 32L
        name <- "xoshiro128plus"
      } else {
        size_int_bits <- 64L
        name <- "xoshiro256plus"
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

    ##' @description Number of streams available
    size = function() {
      private$n_streams
    },

    ##' @description The jump function updates the random number state for
    ##'   each stream by advancing it to a state equivalent to
    ##'   2^128 numbers drawn from each stream.
    jump = function() {
      dust_rng_jump(private$ptr, private$float)
      invisible(self)
    },

    ##' @description Longer than `$jump`, the `$long_jump` method is
    ##'   equivalent to 2^192 numbers drawn from each stream.
    long_jump = function() {
      dust_rng_long_jump(private$ptr, private$float)
      invisible(self)
    },

    ##' @description Generate `n` numbers from a standard uniform distribution
    ##'
    ##' @param n Number of samples to draw (per stream)
    ##'
    ##' @param n_threads Number of threads to use; see Details
    random_real = function(n, n_threads = 1L) {
      dust_rng_random_real(private$ptr, n, n_threads, private$float)
    },

    ##' @description Generate `n` numbers from a standard normal distribution
    ##'
    ##' @param n Number of samples to draw (per stream)
    ##'
    ##' @param n_threads Number of threads to use; see Details
    ##'
    ##' @param algorithm Name of the algorithm to use; currently `box_muller`
    ##'   and `ziggurat` are supported, with the latter being considerably
    ##'   faster.
    random_normal = function(n, n_threads = 1L, algorithm = "box_muller") {
      dust_rng_random_normal(private$ptr, n, n_threads, algorithm,
                             private$float)
    },

    ##' @description Generate `n` numbers from a uniform distribution
    ##'
    ##' @param n Number of samples to draw (per stream)
    ##'
    ##' @param min The minimum of the distribution (length 1 or n)
    ##'
    ##' @param max The maximum of the distribution (length 1 or n)
    ##'
    ##' @param n_threads Number of threads to use; see Details
    uniform = function(n, min, max, n_threads = 1L) {
      dust_rng_uniform(private$ptr, n, min, max, n_threads, private$float)
    },

    ##' @description Generate `n` numbers from a normal distribution
    ##'
    ##' @param n Number of samples to draw (per stream)
    ##'
    ##' @param mean The mean of the distribution (length 1 or n)
    ##'
    ##' @param sd The standard deviation of the distribution (length 1 or n)
    ##'
    ##' @param n_threads Number of threads to use; see Details
    ##'
    ##' @param algorithm Name of the algorithm to use; currently `box_muller`
    ##'   and `ziggurat` are supported, with the latter being considerably
    ##'   faster.
    normal = function(n, mean, sd, n_threads = 1L, algorithm = "box_muller") {
      dust_rng_normal(private$ptr, n, mean, sd, n_threads, algorithm,
                      private$float)
    },

    ##' @description Generate `n` numbers from a binomial distribution
    ##'
    ##' @param n Number of samples to draw (per stream)
    ##'
    ##' @param size The number of trials (zero or more, length 1 or n)
    ##'
    ##' @param prob The probability of success on each trial
    ##'   (between 0 and 1, length 1 or n)
    ##'
    ##' @param n_threads Number of threads to use; see Details
    binomial = function(n, size, prob, n_threads = 1L) {
      dust_rng_binomial(private$ptr, n, size, prob, n_threads, private$float)
    },

    ##' @description Generate `n` numbers from a negative binomial distribution
    ##'
    ##' @param n Number of samples to draw (per stream)
    ##'
    ##' @param size The target number of successful trials
    ##'   (zero or more, length 1 or n)
    ##'
    ##' @param prob The probability of success on each trial
    ##'   (between 0 and 1, length 1 or n)
    ##'
    ##' @param n_threads Number of threads to use; see Details
    nbinomial = function(n, size, prob, n_threads = 1L) {
      dust_rng_nbinomial(private$ptr, n, size, prob, n_threads, private$float)
    },

    ##' @description Generate `n` numbers from a hypergeometric distribution
    ##'
    ##' @param n Number of samples to draw (per stream)
    ##'
    ##' @param n1 The number of white balls in the urn (called n in
    ##'   R's [rhyper])
    ##'
    ##' @param n2 The number of black balls in the urn (called m in
    ##'   R's [rhyper])
    ##'
    ##' @param k The number of balls to draw
    ##'
    ##' @param n_threads Number of threads to use; see Details
    hypergeometric = function(n, n1, n2, k, n_threads = 1L) {
      dust_rng_hypergeometric(private$ptr, n, n1, n2, k, n_threads,
                              private$float)
    },

    ##' @description Generate `n` numbers from a gamma distribution
    ##'
    ##' @param n Number of samples to draw (per stream)
    ##'
    ##' @param shape Shape
    ##'
    ##' @param scale Scale
    ##''
    ##' @param n_threads Number of threads to use; see Details
    gamma = function(n, shape, scale, n_threads = 1L) {
      dust_rng_gamma(private$ptr, n, shape, scale, n_threads,
                              private$float)
    },

    ##' @description Generate `n` numbers from a Poisson distribution
    ##'
    ##' @param n Number of samples to draw (per stream)
    ##'
    ##' @param lambda The mean (zero or more, length 1 or n). Only valid for
    ##' lambda <= 10^7
    ##'
    ##' @param n_threads Number of threads to use; see Details
    poisson = function(n, lambda, n_threads = 1L) {
      dust_rng_poisson(private$ptr, n, lambda, n_threads, private$float)
    },

    ##' @description Generate `n` numbers from a exponential distribution
    ##'
    ##' @param n Number of samples to draw (per stream)
    ##'
    ##' @param rate The rate of the exponential
    ##'
    ##' @param n_threads Number of threads to use; see Details
    exponential = function(n, rate, n_threads = 1L) {
      dust_rng_exponential(private$ptr, n, rate, n_threads, private$float)
    },

    ##' @description Generate `n` draws from a Cauchy distribution.
    ##'
    ##' @param n Number of samples to draw (per stream)
    ##'
    ##' @param location The location of the peak of the distribution
    ##'   (also its median)
    ##'
    ##' @param scale A scale parameter, which specifies the distribution's
    ##'   "half-width at half-maximum"
    ##'
    ##' @param n_threads Number of threads to use; see Details
    cauchy = function(n, location, scale, n_threads = 1L) {
      dust_rng_cauchy(private$ptr, n, location, scale, n_threads, private$float)
    },

    ##' @description Generate `n` draws from a multinomial distribution.
    ##'   In contrast with most of the distributions here, each draw is a
    ##'   *vector* with the same length as `prob`.
    ##'
    ##' @param n The number of samples to draw (per stream)
    ##'
    ##' @param size The number of trials (zero or more, length 1 or n)
    ##'
    ##' @param prob A vector of probabilities for the success of each
    ##'   trial. This does not need to sum to 1 (though all elements
    ##'   must be non-negative), in which case we interpret `prob` as
    ##'   weights and normalise so that they equal 1 before sampling.
    ##'
    ##' @param n_threads Number of threads to use; see Details
    multinomial = function(n, size, prob, n_threads = 1L) {
      dust_rng_multinomial(private$ptr, n, size, prob, n_threads,
                           private$float)
    },

    ##' @description
    ##' Returns the state of the random number stream. This returns a
    ##' raw vector of length 32 * n_streams. It is primarily intended for
    ##' debugging as one cannot (yet) initialise a dust_rng object with this
    ##' state.
    state = function() {
      dust_rng_state(private$ptr, private$float)
    }
  ))
