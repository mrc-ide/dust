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
##' * That class must also include a typedef that describes the
##'   model's floating point type, `real_t`. Most models can include
##'   `typedef double real_t;` in their public section.
##'
##' * The model must have a method `size()` returning `size_t` which
##'   returns the size of the system. This size may depend on values
##'   in your initialisation object but is constant within a model
##'   run.
##'
##' * The model must have a method `initial` (which may not be
##'   `const`), taking a step number (`size_t`) and returning a
##'   `std::vector<real_t>` of initial state for the model.
##'
##' * The model must have a method `update` taking arguments:
##'   - `size_t step`: the step number
##'   - `const double * state`: the state at the beginning of the
##'      step
##'   - `dust::rng_state_t<real_t>& rng_state`: the dust random number
##'     generator state - this *must* be a reference, as it will be modified
##'     as random numbers are drawn
##'   - `double *state_next`: the end state of the model
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
##'   object of type `model::init_t` from an R list.  We use cpp11 for
##'   this.  Your function will look like:
##'
##' ```
##' template <>
##' model::init_t dust_data<model>(cpp11::list data) {
##'   return ...;
##' }
##' ```
##'
##' With the body interacting with `data` to create an object of type
##'   `model::init_t` and returning it.  This function will be called
##'   in serial and may use anything in the cpp11 API.  All elements of
##'   the returned object must be standard C/C++ (e.g., STL) types and
##'   *not* cpp11/R types.
##'
##' Your model *may* provided a template specialisation
##'   `dust_data<model::init_t>()` returning a `cpp11::sexp` for
##'   returning arbitrary information back to the R session:
##'
##' ```
##' template <>
##' cpp11::sexp dust_info<model>(const model::init_t& data) {
##'   return cpp11::wrap(...);
##' }
##' ```
##'
##' What you do with this is up to you. If not present then the
##'   `info()` method on the created object will return `NULL`.
##'   Potential use cases for this are to return information about
##'   variable ordering, or any processing done while accepting the
##'   data object used to create the data fed into the particles.
##'
##' @section Confuring your model:
##'
##' You can optionally use C++ psuedo-attributes to configure the
##'   generated code. Currently we support two attributes:
##'
##' * `[[dust::type(typename)]]` will tell dust the name of your
##'   target C++ class (in this example `typename`). You will need to
##'   use this if your file uses more than a single class, as
##'   otherwise will try to detect this using extremely simple
##'   heuristics.
##'
##' * `[[dust::name(modelname)]]` will tell dust the name of the model
##'   for exporting to R. For technical reasons this must be
##'   alphanumeric characters only (sorry, no underscore) and must not
##'   start with a number. If not included then the C++ type name will
##'   be used (either specified with `[[dust::type]]` or detected).
##'
##' @title Create a dust model from a C++ input file
##'
##' @param filename The path to a single C++ file
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
##' # The model can be compiled and loaded with dust::dust(filename)
##' # but it's faster in this example to use the prebuilt version in
##' # the package
##' model <- dust::dust_example("walk")
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
dust <- function(filename, quiet = FALSE, workdir = NULL) {
  assert_file_exists(filename)
  config <- parse_metadata(filename)
  compile_and_load(filename, config, quiet, workdir)
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
                           all.files = TRUE, no.. = TRUE)),
      file.path("R", dir(file.path(path, "R"),
                           all.files = TRUE, no.. = TRUE)))
    contents <- contents[!grepl(".+\\.(o|so|dll)", contents)]
    allowed <- c("DESCRIPTION", "NAMESPACE", "src", "R",
                 "src/Makevars", "src/dust.cpp", "R/dust.R",
                 "src/cpp11.cpp", "R/cpp11.R",
                 "src/dust.cu", "src/dust.hpp")
    extra <- setdiff(contents, allowed)
    if (length(extra)) {
      stop(sprintf("Path '%s' does not look like a dust directory", path))
    }
  }
  path
}
