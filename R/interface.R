##' Create a dust model from a C++ input file.  This function will
##' compile the dust support around your model and return an object
##' that can be used to work with the model (see the Details below).
##'
##' @section Input requirements:
##'
##' Your input dust model must satisfy a few requirements.
##'
##' * Define some class that implements your model (below `model` is
##'   assumed to be the class name)
##'
##' * That class must define a type `init_t` (so `model::init_t`) that
##'   contains its internal data and the model must be constructable
##'   with a const reference to this type (`const model::init_t&
##'   data`)
##'
##' * That class must also include typedefs that describe the standard
##'   floating point and integer types (`real_t` and `int_t`
##'   respectively). Most models can include `typedef double real_t;`
##'   and `typedef int int_t;` in their public section.
##'
##' * The model must have a method `size()` returning `size_t` which
##'   returns the size of the system. This size may depend on values
##'   in your initialisation object but is constant within a model
##'   run.
##'
##' * The model must have a method `update` (which may not be
##'   `const`), taking a step number (`size_t`) and returning a
##'   `std::vector<real_t>` of initial state for the model.
##'
##' * The model must have a method `update` taking arguments:
##'   - `size_t step`: the step number
##'   - `const std::vector<real_t>& state`: the state at the beginning of the
##'      step
##'   - `dust::RNG<real_t, int_t>& rng`: the dust random number generator
##'   - `std::vector<real_t>& state_next`: the end state of the model
##'     (to be written to by your function)
##'
##' Your `update` function is the core here and should update the
##'   state of the system - you are expected to update all values of
##'   `state` on return.
##'
##' It is very important that none of the functions in the class use
##'   the R API in any way as these functions will be called in
##'   parallel.
##'
##' You must also provide a data-wrangling function for producing an
##'   object of type `model::init_t` from an R list.  We use Rcpp for
##'   this.  Your function will look like:
##'
##' ```
##' template <>
##' model::init_t dust_data<model>(Rcpp::List data) {
##'   return ...;
##' }
##' ```
##'
##' With the body interacting with `data` to create an object of type
##'   `model::init_t` and returning it.  This function will be called
##'   in serial and may use anything in the Rcpp API.  All elements of
##'   the returned object must be standard C/C++ (e.g., STL) types and
##'   *not* Rcpp types.
##'
##' Your model *may* provided a template specialisation
##'   `dust_data<model::init_t>()` returning a `Rcpp::RObject` for
##'   returning arbitrary information back to the R session:
##'
##' ```
##' template <>
##' Rcpp::RObject dust_info<model>(const model::init_t& data) {
##'   return Rcpp::wrap(...);
##' }
##' ```
##'
##' What you do with this is up to you. If not present then the
##'   `info()` method on the created object will return `NULL`.
##'   Potential use cases for this are to return information about
##'   variable ordering, or any processing done while accepting the
##'   data object used to create the data fed into the particles.
##'
##' @title Create a dust model from a C++ input file
##'
##' @param filename The path to a single C++ file
##'
##' @param type The name of the "type" (the C++ class) that represents
##'   your model.  If \code{NULL} we try to work this out from your
##'   file using extremely simple heuristics.
##'
##' @param name The name of the model; for technical reasons this must
##'   be alphanumeric characters only (sorry, no underscore) and must
##'   not start with a number.  If \code{NULL} the value of
##'   \code{type} will be used.
##'
##' @param quiet Logical, indicating if compilation messages from
##'   \code{pkgbuild} should be displayed.  Error messages will be
##'   displayed on compilation failure regardless of the value used.
##'
##' @param workdir Optional working directory to use.  If \code{NULL}
##'   uses a temporary directory.  By using a different directory of
##'   your choosing you can see the generated code.
##'
##' @export
##' @examples
##'
##' # dust includes a couple of very simple examples
##' filename <- system.file("examples/walk.cpp", package = "dust")
##'
##' # This model implements a random walk with a parameter coming from
##' # R representing the standard deviation of the walk
##' writeLines(readLines(filename))
##'
##' # Compile and load the object:
##' model <- dust(filename, quiet = TRUE)
##'
##' # Print the object and you can see the methods that it provides
##' model
##'
##' # Create a model with standard deviation of 1, initial step zero
##' # and 30 particles
##' obj <- model$new(list(sd = 1), 0, 30)
##' obj
##'
##' # Curent state is all zero
##' obj$state()
##'
##' # Current step is also zero
##' obj$step()
##'
##' # Run the model up to step 100
##' obj$run(100)
##'
##' # Reorder/resample the particles:
##' obj$reorder(sample(30, replace = TRUE))
##'
##' # See the state again
##' obj$state()
dust <- function(filename, type = NULL, name = NULL, quiet = FALSE,
                 workdir = NULL) {
  assert_file_exists(filename)
  if (is.null(type)) {
    type <- dust_guess_type(readLines(filename))
  }
  if (is.null(name)) {
    name <- type
  }
  compile_and_load(filename, type, name, quiet, workdir)
}

## NOTE: R6 classes do not support interfaces and inheritence is not
## really needed here as this does not *do* anything, so consider this
## a hack to allow Roxygen's R6 documentation to work.

##' @rdname dust
dust_interface <- R6::R6Class(
  "dust",
  cloneable = FALSE,

  public = list(
    ##' @description
    ##' Create a new model
    ##'
    ##' @param data Data to initialise your model with; a `list`
    ##' object, but the required elements will depend on the details of
    ##' your model.
    ##'
    ##' @param step Initial step - must be nonnegative
    ##'
    ##' @param n_particles Number of particles to create - must be at
    ##' least 1
    ##'
    ##' @param n_threads Number of OMP threads to use, if `dust` and
    ##' your model were compiled with OMP support (details to come)
    ##'
    ##' @param n_generators The number of random number generators to
    ##' use. Must be at least `n_threads` and a multiple of
    ##' `n_threads`.  You can use this to ensure reproducible results
    ##' when changing the number of threads, while preserving
    ##' statistically reasonable random numbers (for example setting
    ##' `n_generators = 64` and then using 1, 2, 4, 8, ..., 64 threads
    ##' as your computational capacity allows).
    ##'
    ##' @param seed Seed to use for the random number generator
    ##' (positive integer)
    initialize = function(data, step, n_particles, n_threads = 1L,
                          n_generators = 1L, seed = 1L) {
    },

    ##' @description
    ##' Run the model up to a point in time, returning the filtered state
    ##' at that point.
    ##'
    ##' @param step_end Step to run to (if less than or equal to the current
    ##' step(),silently nothing will happen)
    run = function(step_end) {
    },

    ##' @description
    ##' Reset the model while preserving the random number stream state
    ##'
    ##' @param data New data for the model (see constructor)
    ##' @param step New initial step for the model (see constructor)
    reset = function(data, step) {
    },

    ##' @description
    ##' Return full model state
    ##' @param index Optional index to select state using
    state = function(index = NULL) {
    },

    ##' @description
    ##' Return current model step
    step = function() {
    },

    ##' @description
    ##' Reorder or resample particles.
    ##' @param index An integer vector, with values between 1 and n_particles,
    ##' indicating the index of the current particles that new particles should
    ##' take.
    reorder = function(index) {
    },

    ##' @description
    ##' Returns information about the data that your model was created with.
    ##' Only returns non-NULL if the model provides a `dust_info` template
    ##' specialisation.
    info = function() {
    }
  ))


dust_class <- function(alloc, run, set_index_y, reset, state, step, reorder,
                       classname = "dust") {
  R6::R6Class(
    classname,
    cloneable = FALSE,

    private = list(
      cpp_alloc = alloc,
      cpp_run = run,
      cpp_set_index_y = set_index_y,
      cpp_reset = reset,
      cpp_state = state,
      cpp_step = step,
      cpp_reorder = reorder,
      data = NULL,
      ptr = NULL
    ),

    public = list(
      initialize = function(data, step, n_particles, n_threads = 1L,
                            n_generators = 1L, seed = 1L) {
        res <- .Call(private$cpp_alloc, data, step, n_particles,
                     n_threads, n_generators, seed)
        private$ptr <- res[[1L]]
        private$data <- res[[2L]]
      },

      run = function(step_end) {
        .Call(private$cpp_run, private$ptr, step_end)
      },

      ## TODO: this probably needs changing as we can't set state and
      ## index until data has been set.
      reset = function(data, step) {
        private$data <- .Call(private$cpp_reset, private$ptr, data, step)
        invisible()
      },

      set_index_y = function(index_y) {
        .Call(private$cpp_set_index_y, private$ptr, index_y)
        invisible()
      },

      state = function(index = NULL) {
        .Call(private$cpp_state, private$ptr, index)
      },

      step = function() {
        .Call(private$cpp_step, private$ptr)
      },

      reorder = function(index) {
        .Call(private$cpp_reorder, private$ptr, as.integer(index))
        invisible()
      },

      info = function() {
        private$data
      }
    ))
}


dust_guess_type <- function(txt) {
  re <- "^\\s*class\\s+([^{ ]+)\\s*(\\{.*|$)"
  i <- grep(re, txt)
  if (length(i) != 1L) {
    stop("Could not automatically detect class name")
  }
  sub(re, "\\1", txt[[i]])
}


dust_workdir <- function(path) {
  if (is.null(path)) {
    path <- tempfile()
  } else if (file.exists(path)) {
    if (!is_directory(path)) {
      stop(sprintf("Path '%s' already exists but is not a directory",
                   path))
    }
    contents <- c(
      dir(path, all.files = TRUE, no.. = TRUE),
      file.path("src", dir(file.path(path, "src"),
                           all.files = TRUE, no.. = TRUE)))
    contents <- contents[!grepl(".+\\.(o|so|dll)", contents)]
    allowed <- c("DESCRIPTION", "NAMESPACE", "src",
                 "src/Makevars", "src/dust.cpp")
    extra <- setdiff(contents, allowed)
    if (length(extra)) {
      stop(sprintf("Path '%s' does not look like a dust directory", path))
    }
  }
  path
}


## Avoid note about "private" not being found and inspection of .Call
## problems; these symbols will be filled in correctly by the factory
## above.
private <- list(cpp_alloc = structure(list(), class = "NativeSymbolInfo"),
                cpp_run = structure(list(), class = "NativeSymbolInfo"),
                cpp_reset = structure(list(), class = "NativeSymbolInfo"),
                cpp_state = structure(list(), class = "NativeSymbolInfo"),
                cpp_step = structure(list(), class = "NativeSymbolInfo"),
                cpp_reorder = structure(list(), class = "NativeSymbolInfo"))
