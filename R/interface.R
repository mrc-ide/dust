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
##' * That class must define a type `internal_type` (so
##'   `model::internal_type`) that contains its internal data that the
##'   model may change during execution (i.e., that is not shared
##'   between particles). If no such data is needed, you can do
##'   `using internal_type = dust::no_internal;` to indicate this.
##'
##' * We also need a type `shared_type` that contains *constant* internal
##'   data is shared between particles (e.g., dimensions, arrays that
##'   are read but not written). If no such data is needed, you can do
##'   `using share_type = dust::no_shared;` to indicate this.
##'
##' * That class must also include a type alias that describes the
##'   model's floating point type, `real_type`. Most models can include
##'   `using real_type = double;` in their public section.
##'
##' * The class must also include a type alias that describes the model's
##'   *data* type. If your model does not support data, then write
##'   `using data_type = dust::no_data;`, which disables the
##'   `compare_data` and `set_data` methods.  Otherwise see
##'   `vignette("data")` for more information.
##'
##' * The class must have a constructor that accepts `const
##'   dust::pars_type<model>& pars` for your type `model`. This will have
##'   elements `shared` and `internal` which you can assign into your
##'   model if needed.
##'
##' * The model must have a method `size()` returning `size_t` which
##'   returns the size of the system. This size may depend on values
##'   in your initialisation object but is constant within a model
##'   run.
##'
##' * The model must have a method `initial` (which may not be
##'   `const`), taking a time step number (`size_t`) and returning a
##'   `std::vector<real_type>` of initial state for the model.
##'
##' * The model must have a method `update` taking arguments:
##'   - `size_t time`: the time step number
##'   - `const double * state`: the state at the beginning of the
##'      time step
##'   - `dust::rng_state_type<real_type>& rng_state`: the dust random number
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
##'   producing an object of type `dust::pars_type<model>` from an R list.  We
##'   use cpp11 for this.  Your function will look like:
##'
##' ```
##' namespace dust {
##' template <>
##' dust::pars_type<model> dust_pars<model>(cpp11::list pars) {
##'   // ...
##'   return dust::pars_type<model>(shared, internal);
##' }
##' }
##' ```
##'
##' With the body interacting with `pars` to create an object of type
##'   `model::shared_type` and `model::internal_type` before returning the
##'   `dust::pars_type` object.  This function will be called in serial
##'   and may use anything in the cpp11 API.  All elements of the
##'   returned object must be standard C/C++ (e.g., STL) types and
##'   *not* cpp11/R types. If your model uses only shared or internal,
##'   you may use the single-argument constructor overload to
##'   `dust::pars_type` which is equivalent to using `dust::no_shared` or
##'   `dust::no_internal` for the missing argument.
##'
##' Your model *may* provided a template specialisation
##'   `dust::dust_info<model>()` returning a `cpp11::sexp` for
##'   returning arbitrary information back to the R session:
##'
##' ```
##' namespace dust {
##' template <>
##' cpp11::sexp dust_info<model>(const dust::pars_type<sir>& pars) {
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
##' Your model should only throw exceptions as a last resort. One such
##'   last resort exists already if `rbinom` is given invalid inputs
##'   to prevent an infinite loop. If an error is thrown, all
##'   particles will complete their current run, and then the error
##'   will be rethrown - this is required by our parallel processing
##'   design. Once this happens though the state of the system is
##'   "inconsistent" as it contains particles that have run for
##'   different lengths of time. You can extract the state of the
##'   system at the point of failure (which may help with debugging)
##'   but you will be unable to continue running the object until
##'   either you reset it (with `$update_state()`). An error will be
##'   thrown otherwise.
##'
##' Things are worse on a GPU; if an error is thrown by the RNG code
##'   (happens in `rbinom` when given impossible inputs such as
##'   negative sizes, probabilities less than zero or greater than 1)
##'   then we currently use CUDA's `__trap()` function which will
##'   require a process restart to be able to use anything that uses
##'   the GPU again, covering all methods in the class.  However, this
##'   is preferable to the infinite loop that would otherwise be
##'   caused.
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
##' @param real_type Optionally, a string indicating a substitute type to
##'   swap in for your model's `real_type` declaration. If given, then we
##'   replace the string `using real_type = (double|float)` with the
##'   given type. This is primarily intended to be used as `gpu =
##'   TRUE, real_type = "float"` in order to create model for the GPU
##'   that will use 32 bit `floats` (rather than 64 bit doubles, which
##'   are much slower). For CPU models decreasing precision of your
##'   real type will typically just decrease precision for no
##'   additional performance.
##'
##' @param linking_to Optionally, a character vector of additional
##'   packages to add to the `DESCRIPTION`'s `LinkingTo` field. Use
##'   this when your model pulls in C++ code that is packaged within
##'   another package's header-only library.
##'
##' @param cpp_std The C++ standard to use. This will be be set into
##'   the `DESCRIPTION` of the package as the `SystemRequirements`
##'   field. Sensible options are `C++11`, `C++14` etc. See the
##'   section "Using C++ code" in "Writing R extensions". The minimum
##'   allowed version is C++11 but R supports much more recent
##'   versions now (especially more recent versions of R).
##'
##' @param skip_cache Logical, indicating if the cache of previously
##'   compiled models should be skipped. If `TRUE` then your model will
##'   not be looked for in the cache, nor will it be added to the
##'   cache after compilation.
##'
##' @seealso [`dust::dust_generator`] for a description of the class
##'   of created objects, and [dust::dust_example()] for some
##'   pre-built examples. If you want to just generate the code and
##'   load it yourself with `pkgload::load_all` or some other means,
##'   see [`dust::dust_generate`])
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
##' # Create a model with standard deviation of 1, initial time step zero
##' # and 30 particles
##' obj <- model$new(list(sd = 1), 0, 30)
##' obj
##'
##' # Curent state is all zero
##' obj$state()
##'
##' # Current time is also zero
##' obj$time()
##'
##' # Run the model up to time step 100
##' obj$run(100)
##'
##' # Reorder/resample the particles:
##' obj$reorder(sample(30, replace = TRUE))
##'
##' # See the state again
##' obj$state()
dust <- function(filename, quiet = FALSE, workdir = NULL, gpu = FALSE,
                 real_type = NULL, linking_to = NULL, cpp_std = NULL,
                 skip_cache = FALSE) {
  filename <- dust_prepare(filename, real_type)
  compile_and_load(filename, quiet, workdir, cuda_check(gpu), linking_to,
                   cpp_std, skip_cache)
}


##' Generate a package out of a dust model. The resulting package can
##' be installed or loaded via `pkgload::load_all()` though it
##' contains minimal metadata and if you want to create a persistent
##' package you should use [dust::dust_package()].  This function is
##' intended for cases where you either want to inspect the code or
##' generate it once and load multiple times (useful in some workflows
##' with CUDA models).
##'
##' @title Generate dust code
##'
##' @inheritParams dust
##'
##' @param mangle Logical, indicating if the model name should be
##'   mangled when creating the package. This is safer if you will
##'   load multiple copies of the package into a single session, but
##'   is `FALSE` by default as the generated code is easier to read.
##'
##' @return The path to the generated package (will be `workdir` if
##'   that was provided, otherwise a temporary directory).
##'
##' @export
##' @examples
##' filename <- system.file("examples/walk.cpp", package = "dust")
##' path <- dust::dust_generate(filename)
##'
##' # Simple package created:
##' dir(path)
##' dir(file.path(path, "R"))
##' dir(file.path(path, "src"))
dust_generate <- function(filename, quiet = FALSE, workdir = NULL, gpu = FALSE,
                          real_type = NULL, linking_to = NULL, cpp_std = NULL,
                          mangle = FALSE) {
  filename <- dust_prepare(filename, real_type)
  skip_cache <- TRUE
  res <- generate_dust(filename, quiet, workdir, cuda_check(gpu), linking_to,
                       cpp_std, skip_cache, mangle)
  cpp11::cpp_register(res$path, quiet = quiet)
  res$path
}


dust_prepare <- function(filename, real_type) {
  assert_file_exists(filename)
  if (!is.null(real_type)) {
    filename <- dust_rewrite_real(filename, real_type)
  }
  filename
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


dust_rewrite_real <- function(filename, real_type) {
  dest <- tempfile(fileext = ".cpp")
  re1 <- "^(\\s*typedef\\s+)([_a-zA-Z0-9]+)(\\s+real_type.*)"
  re2 <- "^(\\s*using\\s+real_type\\s*=\\s*)([_a-zA-Z0-9]+)(;.*)"
  code <- readLines(filename)
  i1 <- grep(re1, code)
  i2 <- grep(re2, code)
  if (length(i1) == 0 && length(i2) == 0) {
    stop(sprintf("did not find real_type declaration in '%s'", filename))
  }
  code[i1] <- sub(re1, sprintf("\\1%s\\3", real_type), code[i1])
  code[i2] <- sub(re2, sprintf("\\1%s\\3", real_type), code[i2])
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
