##' Updates a dust model in a package. The user-provided code is
##' assumed to the in `inst/dust` as a series of C++ files; a file
##' `inst/dust/model.cpp` will be transformed into a file
##' `src/model.cpp`.
##'
##' If your code provides a class `model` then dust will create C++
##' functions such as `dust_model_alloc` - if your code also includes
##' names such as this, compilation will fail due to duplicate
##' symbols.
##'
##' We add "cpp11 attributes" to the created functions, and will run
##' [cpp11::cpp_register()] on them once the generated code
##' has been created.
##'
##' Your package needs a `src/Makevars` file to enable openmp (if your
##' system supports it). If it is not present then a suitable Makevars
##' will be written, containing
##'
##' ```
##' PKG_CXXFLAGS=$(SHLIB_OPENMP_CXXFLAGS)
##' PKG_LIBS=$(SHLIB_OPENMP_CXXFLAGS)
##' ```
##'
##' following "Writing R Extensions" (see section "OpenMP support").
##' If your package does contain a `src/Makevars` file we do not
##' attempt to edit it but will error if it looks like it does not
##' contain these lines or similar.
##'
##' You also need to make sure that your package loads the dynamic
##' library; if you are using roxygen, then you might create a file
##' (say, `R/zzz.R`) containing
##'
##' ```
##' #' @useDynLib packagename, .registration = TRUE
##' NULL
##' ```
##'
##' substituting `packagename` for your package name as
##' appropriate. This will create an entry in `NAMESPACE`.
##'
##' @param path Path to the package
##'
##' @param quiet Passed to `cpp11::cpp_register`, if `TRUE` suppresses
##'   informational notices about updates to the cpp11 files
##'
##' @title Create dust model in package
##'
##' @return Nothing, this function is called for its side effects
##'
##' @seealso `vignette("dust")` which contains more discussion of this
##'   process
##'
##' @export
##'
##' @examples
##' # This is explained in more detail in the package vignette
##' path <- system.file("examples/sir.cpp", package = "dust", mustWork = TRUE)
##' dest <- tempfile()
##' dir.create(dest)
##' dir.create(file.path(dest, "inst/dust"), FALSE, TRUE)
##' writeLines(c("Package: example",
##'              "Version: 0.0.1",
##'              "LinkingTo: cpp11, dust"),
##'            file.path(dest, "DESCRIPTION"))
##' writeLines("useDynLib('example', .registration = TRUE)",
##'            file.path(dest, "NAMESPACE"))
##' file.copy(path, file.path(dest, "inst/dust"))
##'
##' # An absolutely minimal skeleton contains a DESCRIPTION, NAMESPACE
##' # and one or more dust model files to compile:
##' dir(dest, recursive = TRUE)
##'
##' # Running dust_package will fill in the rest
##' dust::dust_package(dest)
##'
##' # More files here now
##' dir(dest, recursive = TRUE)
dust_package <- function(path, quiet = FALSE) {
  ## 1. check that the package is legit
  root <- package_validate(path)
  path_dust <- file.path(root, "inst/dust")
  path_src <- file.path(root, "src")
  path_r <- file.path(root, "R")

  ## 2. find target model files

  ## TODO: support alternative location, perhaps not in package - that
  ## will suit odin.
  files <- dir(path_dust, pattern = "\\.cpp$")
  if (length(files) == 0L) {
    stop(sprintf("No dust files found in '%s/inst/dust'", root))
  }

  ## 3. identify destination src files, validate signature
  package_validate_destination(root, files)

  ## 4. generate code
  data <- lapply(file.path(path_dust, files), package_generate)

  dir.create(path_src, FALSE, TRUE)
  dir.create(path_r, FALSE, TRUE)

  for (d in data) {
    for (p in names(d$src)) {
      writelines_if_changed(
        c(dust_header("//"), d$src[[p]]), file.path(path_src, p))
    }
  }

  code_r <- c(dust_header("##"), vcapply(data, "[[", "r"))
  writelines_if_changed(code_r, file.path(path_r, "dust.R"))

  pkg_makevars <- file.path(path, "src/Makevars")
  if (file.exists(pkg_makevars)) {
    package_validate_makevars_openmp(read_lines(pkg_makevars))
  } else {
    writelines_if_changed(
      read_lines(dust_file("template/Makevars.pkg")), pkg_makevars)
  }

  ## 5. compile attributes
  cpp11::cpp_register(path, quiet = quiet)

  ## 6. return path, invisibly
  invisible(path)
}


package_validate <- function(path) {
  paths <- c("DESCRIPTION", "NAMESPACE")
  for (p in paths) {
    if (!file.exists(file.path(path, p))) {
      stop(sprintf("Expected a file '%s' at path '%s'", p, path))
    }
  }

  desc <- pkgload::pkg_desc(path)
  deps <- desc$get_deps()
  package_validate_has_dep(deps, "cpp11", "LinkingTo")
  package_validate_has_dep(deps, "dust", "LinkingTo")

  name <- desc$get_field("Package")
  if (grepl("[._]+", name)) {
    stop(sprintf(
      "Package name must not contain '.' or '_' (found '%s')", name))
  }
  package_validate_namespace(file.path(path, "NAMESPACE"), name)

  path
}


## NOTE: Should be able to do this directly with desc but there is a
## small bug: https://github.com/r-lib/desc/pull/97
package_validate_has_dep <- function(deps, package, type) {
  if (!any(deps$package == package & deps$type == type)) {
    stop(sprintf("Expected package '%s' as '%s' in DESCRIPTION",
                 package, type))
  }
}


package_validate_destination <- function(path, files) {
  check <- c(file.path(path, "src", files),
             file.path(path, "R", "dust.R"))
  for (f in check[file.exists(check)]) {
    if (!isTRUE(grepl("^(//|##) Generated by dust", readLines(f, 1)))) {
      stop(sprintf(
        "File '%s' does not look like it was created by dust - stopping",
        f))
    }
  }
}


package_validate_namespace <- function(path, name) {
  exprs <- as.list(parse(path))
  package_validate_namespace_usedynlib(exprs, name)
}


package_validate_namespace_usedynlib <- function(exprs, name) {
  for (e in exprs) {
    if (is_call(e, "useDynLib")) {
      lib <- e[[2]]
      if (is.name(lib)) {
        lib <- deparse(lib)
      }
      if (identical(lib, name)) {
        return()
      }
      stop(sprintf("Found a useDynLib call but not for '%s'", name))
    }
  }
  stop("Did not find a useDynLib call in NAMESPACE")
}


package_generate <- function(filename) {
  config <- parse_metadata(filename)
  model <- read_lines(filename)
  data <- dust_template_data(model, config, NULL)

  template_r <- readLines(dust_file("template/dust.R.template"))
  ## Drop all the roxygen comments here before writing out the R
  ## code. The reasoning here is that we have no way of tying this to
  ## the correct help page, and the user may not be using roxygen at
  ## all, and they could just link to the help file ?dust to get the
  ## same documentation.
  template_r <- paste(template_r[!grepl("\\s*#+'", template_r)],
                      collapse = "\n")
  code_r <- glue_whisker(template_r, data)

  template_hpp <- read_lines(dust_file("template/dust.hpp"))
  template_cpp <- read_lines(dust_file("template/dust.cpp"))
  code_hpp <- glue_whisker(template_hpp, data)
  code_cpp <- glue_whisker(template_cpp, data)
  src <- set_names(paste(code_hpp, code_cpp, sep = "\n\n"), basename(filename))
  list(src = src, r = code_r)
}


package_validate_makevars_openmp <- function(text) {
  ok <- grepl("PKG_CXXFLAGS", text) &&
    grepl("PKG_LIBS", text) &&
    grepl("SHLIB_OPENMP_CXXFLAGS", text)
  if (!ok) {
    stop("Package has a 'src/Makevars' but no openmp flags support")
  }
}
