##' Return information about OpenMP support for this system.  For
##' individual models look at the `$has_openmp()` method
##'
##' @title Information about OpenMP support
##'
##' @param check_compile Logical, indicating if we should check if we
##'   can compile an openmp program - this is slow the first time.
##'
##' @seealso [dust_openmp_threads()] for setting a polite number of
##'   threads.
##'
##' @export
##' @examples
##' # System wide support
##' dust::dust_openmp_support()
##'
##' # Support compiled into a generator
##' walk <- dust::dust_example("walk")
##' walk$public_methods$has_openmp()
##'
##' # Support from an instance of that model
##' model <- walk$new(list(sd = 1), 0, 1)
##' model$has_openmp()
dust_openmp_support <- function(check_compile = FALSE) {
  info <- openmp_info()
  if (check_compile) {
    info$has_openmp_compiler <- has_openmp_compiler()
  }
  info
}


##' Politely select a number of threads to use. See Details for the algorithm
##'
##' There are two limits and we will take the smaller of the two.
##'
##' The first limit comes from piggy-backing off of R's normal
##' parallel configuration; we will use the `MC_CORES` environment
##' variable and `mc.cores` option as a guide to how many cores you
##' are happy to use. We take `mc.cores` first, then `MC_CORES`, which
##' is the same behaviour as `parallel::mclapply` and friends.
##'
##' The second limit comes from openmp. If you do not have OpenMP
##' support, then we use one thread (higher numbers have no effect at
##' all in this case). If you do have OpenMP support, we take the
##' smallest of the number of "processors" (reported by
##' `omp_get_num_procs()`) the "max threads" (reported by
##' `omp_get_max_threads()` and "thread_limit" (reported by
##' `omp_get_thread_limit()`.
##'
##' See [dust::dust_openmp_support()] for the values of all the values
##' that go into this calculation.
##'
##' @title Select number of threads
##'
##' @param n Either `NULL` (select automatically) or an integer as
##'   your proposed number of threads.
##'
##' @param action An action to perform if `n` exceeds the maximum
##'   number of threads you can use. Options are "error" (the default,
##'   throw an error), "fix" (print a message and reduce `n` down to
##'   the limit) or "message" (print a message and continue anyway)
##'
##' @return An integer, indicating the number of threads that you can use
##' @export
##' @examples
##' # Default number of threads; tries to pick something friendly,
##' # erring on the conservative side.
##' dust::dust_openmp_threads(NULL)
##'
##' # Try to pick something silly and it will be reduced for you
##' dust::dust_openmp_threads(1000, action = "fix")
dust_openmp_threads <- function(n = NULL, action = "error") {
  info <- openmp_info()
  if (is.null(n)) {
    n <- info$limit
  } else  {
    n <- openmp_check_limit(n, info$limit, action)
  }
  n
}


has_openmp_compiler <- function() {
  if (is.null(cache$has_openmp_compiler)) {
    cache$has_openmp_compiler <- has_openmp_compiler_test()
  }
  cache$has_openmp_compiler
}


## This test uses the 'parallel' example, which as its update() method
## returns the thread number by running omp_get_thread_num()
has_openmp_compiler_test <- function() {
  tryCatch({
    gen <- dust(dust_file("examples/parallel.cpp"), quiet = TRUE)
    mod <- gen$new(list(sd = 1), 0, 1)
    mod$run(1)
    mod$state(2L) == 0
  }, error = function(e) FALSE)
}


## NOTE: This does not return if the *compiler* supports openmp, just
## the runtime.  While we are testing that will be the same thing, but
## after installation from binary this requires really a compile time
## test of a simple openmp program.
openmp_info <- function() {
  env <- Sys.getenv(c("OMP_THREAD_LIMIT", "OMP_NUM_THREADS", "MC_CORES"))
  env <- set_names(as.list(as.integer(env)), names(env))
  info <- cpp_openmp_info()
  info[["mc.cores"]] <- getOption("mc.cores", NA_integer_)

  limit <- list()
  limit$limit_r <- getOption("mc.cores", as.integer(Sys.getenv("MC_CORES", 1)))
  limit$limit_openmp <- min(info$num_procs,
                            info$num_threads,
                            info$thread_limit)
  if (!info$has_openmp) {
    limit$limit_openmp <- 1L
  }
  limit$limit <- min(limit$limit_r, limit$limit_openmp)

  c(info, env, limit)
}


openmp_check_limit <- function(n, limit, action) {
  valid <- c("error", "message", "fix")
  if (!(action %in%  valid)) {
    stop("'action' must be one of ",
         paste(sprintf("'%s'", valid), collapse = ", "))
  }

  if (n > limit) {
    msg <- paste(
      sprintf("Requested number of threads '%d' exceeds a limit of '%d'\n",
              n, limit),
      "See dust::dust_openmp_threads() for details")
    if (action == "error") {
      stop(msg, call. = FALSE)
    } else {
      message(msg)
      if (action == "fix") {
        n <- limit
      }
    }
  }
  n
}
