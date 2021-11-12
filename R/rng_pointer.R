##' @title Create pointer to random number generator
##'
##' For external use, vignette coming soon.
##'
##' @export
##' @examples
##' dust::dust_rng_pointer$new()
dust_rng_pointer <- R6::R6Class(
  "dust_rng_pointer",
  cloneable = FALSE,

  private = list(
    ptr_ = NULL,
    state_ = NULL,
    is_current_ = NULL
  ),

  public = list(
    ##' @field algorithm The name of the generator algorithm used (read-only)
    algorithm = NULL,

    ##' @field n_streams The number of streams of random numbers provided
    ##'   (read-only)
    n_streams = NULL,

    ##' @description Create a new `dust_rng_pointer` object
    ##'
    ##' @param seed The random number seed to use (see [dust::dust_rng]
    ##'   for details)
    ##'
    ##' @param n_streams The number of independent random number streams to
    ##'   create
    ##'
    ##' @param algorithm The random number algorithm to use. The default is
    ##'   `xoshiro256plus` which is a good general choice
    initialize = function(seed = NULL, n_streams = 1L,
                          algorithm = "xoshiro256plus") {
      dat <- dust_rng_pointer_init(n_streams, seed, algorithm)
      private$ptr_ <- dat[[1L]]
      private$state_ <- dat[[2L]]
      private$is_current_ <- TRUE

      self$algorithm <- algorithm
      self$n_streams <- n_streams
      lockBinding("algorithm", self)
      lockBinding("n_streams", self)
    },

    ##' @description Synchronise the R copy of the random number state.
    ##' Typically this is only needed before serialisation if you have
    ##' ever used the object.
    sync = function() {
      dust_rng_pointer_sync(private, self$algorithm)
    },

    ##' @description Return a raw vector of state. This can be used to
    ##' create other generators with the same state.
    state = function() {
      private$state_
    },

    ##' @description Return a logical, indicating if the random number
    ##' state that would be returned by `state()` is "current" (i.e., the
    ##' same as the copy held in the pointer) or not. This is `TRUE` on
    ##' creation or immediately after calling `$sync()` and `FALSE` after
    ##' any use of the pointer.
    is_current = function() {
      private$is_current_
    }
  ))
