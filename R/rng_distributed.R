##' Create a set of initial random number seeds suitable for using
##' within a distributed context (over multiple processes or nodes) at
##' a level higher than a single group of synchronised threads.
##'
##' See `vignette("rng_distributed")` for a proper introduction to
##' these functions.
##'
##' @title Create a set of distributed seeds
##'
##' @param seed Initial seed to use. As for [dust::dust_rng], this can
##'   be `NULL` (create a seed using R's generators), an integer or a
##'   raw vector of appropriate length.
##'
##' @param n_streams The number of streams to create per node. If
##'   passing the results of this seed to a dust object's initialiser
##'   (see [dust::dust_generator]) you can safely leave this at 1, but
##'   if using in a standalone setting, and especially if using
##'   `dust_rng_distributed_pointer`, you may need to set this to the
##'   appropriate length.
##'
##' @param n_nodes The number of separate seeds to create. Each will
##'   be separated by a "long jump" for your generator.
##'
##' @param algorithm The name of an algorithm to use. Alternatively
##'   pass a `dust_generator` or `dust` object here to select the
##'   algorithm used by that object automatically.
##'
##' @return A list of either raw vectors (for
##'   `dust_rng_distributed_state`) or of [dust::dust_rng_pointer]
##'   objects (for `dust_rng_distributed_pointer`)
##'
##' @export
##' @rdname dust_rng_distributed
##' @examples
##' dust::dust_rng_distributed_state(n_nodes = 2)
##' dust::dust_rng_distributed_pointer(n_nodes = 2)
dust_rng_distributed_state <- function(seed = NULL, n_streams = 1L,
                                       n_nodes = 1L,
                                       algorithm = "xoshiro256plus") {
  algorithm <- check_algorithm(algorithm)
  p <- dust_rng_pointer$new(seed, n_streams, algorithm = algorithm)

  ret <- vector("list", n_nodes)
  for (i in seq_len(n_nodes)) {
    s <- p$state()
    ret[[i]] <- s
    if (i < n_nodes) {
      p <- dust_rng_pointer$new(s, n_streams, 1L, algorithm = algorithm)
    }
  }

  ret
}


##' @export
##' @rdname dust_rng_distributed
dust_rng_distributed_pointer <- function(seed = NULL, n_streams = 1L,
                                         n_nodes = 1L,
                                         algorithm = "xoshiro256plus") {
  algorithm <- check_algorithm(algorithm)
  state <- dust_rng_distributed_state(seed, n_streams, n_nodes, algorithm)
  lapply(state, dust_rng_pointer$new,
         n_streams = n_streams, algorithm = algorithm)
}


check_algorithm <- function(algorithm) {
  if (inherits(algorithm, "dust_generator")) {
    algorithm <- algorithm$public_methods$rng_algorithm()
  } else if (inherits(algorithm, "dust")) {
    algorithm <- algorithm$rng_algorithm()
  }
  algorithm
}
