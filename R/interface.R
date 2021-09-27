##' Create a dust model from a C++ input file.  This function will
##' compile the dust support around your model and return an object
##' that can be used to work with the model (see the Details below,
##' and [`dust::dust_generator`]).
##'
##' @section Input requirements:
##'
##' Your input dust model must satisfy a few requirements.
##'
##' * Define some class that implements your model (below `model` is
##'   assumed to be the class name)
##'
##' * That class must define a type `internal_t` (so
##'   `model::internal_t`) that contains its internal data that the
##'   model may change during execution (i.e., that is not shared
##'   between particles). If no such data is needed, you can do
##'   `typedef dust::no_internal internal_t` to indicate this.
##'
##' * We also need a type `shared_t` that contains *constant* internal
##'   data is shared between particles (e.g., dimensions, arrays that
##'   are read but not written). If no such data is needed, you can do
##'   `typedef dust::no_shared shared_t` to indicate this.
##'
##' * That class must also include a typedef that describes the
##'   model's floating point type, `real_t`. Most models can include
##'   `typedef double real_t;` in their public section.
##'
##' * The class must also include a typedef that describes the model's
##'   *data* type. This interface is subject to change, and for now
##'   you should include `typedef dust::no_data data_t` which marks
##'   your class as not supporting data, which disables the
##'   `compare_data` and `set_data` methods.
##'
##' * The class must have a constructor that accepts `const
##'   dust::pars_t<model>& pars` for your type `model`. This will have
##'   elements `shared` and `internal` which you can assign into your
##'   model if needed.
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
##' You must also provide a data/parameter-wrangling function for
##'   producing an object of type `dust::pars_t<model>` from an R list.  We
##'   use cpp11 for this.  Your function will look like:
##'
##' ```
##' namespace dust {
##' template <>
##' dust::pars_t<model> dust_pars<model>(cpp11::list pars) {
##'   // ...
##'   return dust::pars_t<model>(shared, internal);
##' }
##' }
##' ```
##'
##' With the body interacting with `pars` to create an object of type
##'   `model::shared_t` and `model::internal_t` before returning the
##'   `dust::pars_t` object.  This function will be called in serial
##'   and may use anything in the cpp11 API.  All elements of the
##'   returned object must be standard C/C++ (e.g., STL) types and
##'   *not* cpp11/R types. If your model uses only shared or internal,
##'   you may use the single-argument constructor overload to
##'   `dust::pars_t` which is equivalent to using `dust::no_shared` or
##'   `dust::no_internal` for the missing argument.
##'
##' Your model *may* provided a template specialisation
##'   `dust::dust_info<model>()` returning a `cpp11::sexp` for
##'   returning arbitrary information back to the R session:
##'
##' ```
##' namespace dust {
##' template <>
##' cpp11::sexp dust_info<model>(const dust::pars_t<sir>& pars) {
##'   return cpp11::wrap(...);
##' }
##' }
##' ```
##'
##' What you do with this is up to you. If not present then the
##'   `info()` method on the created object will return `NULL`.
##'   Potential use cases for this are to return information about
##'   variable ordering, or any processing done while accepting the
##'   pars object used to create the pars fed into the particles.
##'
##' @section Configuring your model:
##'
##' You can optionally use C++ pseudo-attributes to configure the
##'   generated code. Currently we support two attributes:
##'
##' * `[[dust::class(classname)]]` will tell dust the name of your
##'   target C++ class (in this example `classname`). You will need to
##'   use this if your file uses more than a single class, as
##'   otherwise will try to detect this using extremely simple
##'   heuristics.
##'
##' * `[[dust::name(modelname)]]` will tell dust the name to use for
##'   the class in R code. For technical reasons this must be
##'   alphanumeric characters only (sorry, no underscore) and must not
##'   start with a number. If not included then the C++ type name will
##'   be used (either specified with `[[dust::class()]]` or detected).
##'
##' @section Error handling:
##'
##' Your model throw exceptions as a last resort. One such last resort
##'   exists if `rbinom` is given invalid inputs to prevent an
##'   infinite loop. If an error is thrown, all particles will
##'   complete their current run, and then the error will be rethrown
##'   - this is required by our parallel processing design. Once this
##'   happens though the state of the system is "inconsistent" as it
##'   contains particles that have run for different lengths of
##'   time. You can extract the state of the system at the point of
##'   failure (which may help with debugging) but you will be unable
##'   to continue running the object until either you reset it (with
##'   `$reset()`) or set *both* the state and step with
##'   `$set_state()`. An error will be thrown otherwise.
##'
##' Things are worse on a GPU; if an error is thrown by the RNG code
##'   (happens in `rbinom` when given impossible inputs such as
##'   negative sizes, probabilities less than zero or greater than 1)
##'   then we currently use CUDA's `__trap()` function which will
##'   require a process restart to be able to use any device function
##'   again, covering all methods in the class.  However, this is
##'   preferable to the infinite loop that would otherwise be caused.
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
##' @param gpu Logical, indicating if we should generate GPU
##'   code. This requires a considerable amount of additional software
##'   installed (CUDA toolkit and drivers) as well as a
##'   CUDA-compatible GPU. If `TRUE`, then we call
##'   [dust::dust_cuda_options] with no arguments. Alternatively, call
##'   that function and pass the value here (e.g, `gpu =
##'   dust::dust_cuda_options(debug = TRUE)`). Note that due to the
##'   use of the `__syncwarp()` primitive this may require a GPU with
##'   compute version 70 or higher.
##'
##' @param real_t Optionally, a string indicating a substitute type to
##'   swap in for your model's `real_t` declaration. If given, then we
##'   replace the string `typedef (double|float) real_t` with the
##'   given type. This is primarily intended to be used as `gpu =
##'   TRUE, real_t = "float"` in order to create model for the GPU
##'   that will use 32 bit `floats` (rather than 64 bit doubles, which
##'   are much slower). For CPU models decreasing precision of your
##'   real type will typically just decrease precision for no
##'   additional performance.
##'
##' @seealso [`dust::dust_generator`] for a description of the class of created
##'   objects, and [dust::dust_example()] for some pre-built examples.
##'
##' @return A [`dust::dust_generator`] object based on your source files
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
dust <- function(filename, quiet = FALSE, workdir = NULL, gpu = FALSE,
                 real_t = NULL) {
  assert_file_exists(filename)
  if (!is.null(real_t)) {
    filename <- dust_rewrite_real(filename, real_t)
  }
  compile_and_load(filename, quiet, workdir, cuda_check(gpu))
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


dust_rewrite_real <- function(filename, real_t) {
  dest <- tempfile(fileext = ".cpp")
  re <- "^(\\s*typedef\\s+)([_a-zA-Z0-9]+)(\\s+real_t.*)"
  code <- readLines(filename)
  i <- grep(re, code)
  if (length(i) == 0) {
    stop(sprintf("did not find real_t declaration in '%s'", filename))
  }
  code[i] <- sub(re, sprintf("\\1%s\\3", real_t), code[i])
  writeLines(code, dest)
  dest
}


##' @export
##' @importFrom stats coef
coef.dust <- function(object, ...) {
  object$param()
}


##' @export
coef.dust_generator <- function(object, ...) {
  object$private_fields$param_
}
