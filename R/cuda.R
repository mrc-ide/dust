##' Detect CUDA configuration. This function tries to compile a small
##' program with `nvcc` and confirms that this can be loaded into R,
##' then uses that program to query the presence and capabilities of
##' your NVIDIA GPUs. If this works, then you can use the GPU-enabled
##' dust features, and the information returned will help us.  It's
##' quite slow to execute (several seconds) so we cache the value
##' within a session.  Later versions of dust will cache this across
##' sessions too.
##'
##' Not all installations leave the CUDA libraries on the default
##' paths, and you may need to provide it. Specifically, when we link
##' the dynamic library, if the linker complains about not being able
##' to find `libcudart` then your CUDA libraries are not in the
##' default location. You can manually pass in the `path_cuda_lib`
##' argument, or set the `DUST_PATH_CUDA_LIB` environment variable (in
##' that order of precedence).
##'
##' If you are using older CUDA (< 11.0.0) then you need to provide
##' [CUB](https://nvlabs.github.io/cub/) headers, which we use to
##' manage state on the device (these are included in CUDA 11.0.0 and
##' higher). You can provide this as:
##'
##' * a path to this function (`path_cub_include`)
##' * the environment variable DUST_PATH_CUB_INCLUDE
##' * CUB headers installed into the default location (R >= 4.0.0,
##'   see below).
##'
##' These are checked in turn with the first found taking
##' precedence. The default location is stored with
##' `tools::R_user_dir("dust", "data")`, but this functionality is
##' only available on R >= 4.0.0.
##'
##' To install CUB you can do:
##'
##' ```
##' dust:::cuda_install_cub(NULL)
##' ```
##'
##' which will install CUB into the default path (provide a path on
##' older versions of R and set this path as DUST_PATH_CUB_INCLUDE).
##'
##' For editing your .Renviron file to set these environment
##' variables, `usethis::edit_r_environ()` is very helpful.
##'
##' @title Detect CUDA configuration
##'
##' @param path_cuda_lib Optional path to the CUDA libraries, if they
##'   are not on system library paths. This will be added as
##'   `-L{path_cuda_lib}` in calls to `nvcc`
##'
##' @param path_cub_include Optional path to the CUB headers, if using
##'   CUDA < 11.0.0. See Details
##'
##' @param quiet Logical, indicating if compilation of test programs
##'   should be quiet
##'
##' @param forget Logical, indicating if we should forget cached
##'   values and recompute the configuration
##'
##' @return A list of configuration information. This includes:
##'
##' * `has_cuda`: logical, indicating if it is possible to compile CUDA on
##'   this machine (not necessarily use it though)
##' * `cuda_version`: the version of CUDA found
##' * `devices`: a data.frame of device information:
##'   - id: the device id (integer, typically in a sequence from 0)
##'   - name: the human-friendly name of the device
##'   - memory: the memory of the device, in MB
##'   - version: the compute version for this device
##' * `path_cuda_lib`: path to CUDA libraries, if required
##' * `path_cub_include`: path to CUB headers, if required
##'
##' If compilation of the test program fails, then `has_cuda` will be
##'   `FALSE` and all other elements will be `NULL`.
##'
##' @seealso [dust::dust_cuda_options] which controls additional CUDA
##'   compilation options (e.g., profiling, debug mode or custom
##'   flags)
##'
##' @export
##' @examples
##' # If you have your CUDA library in an unusual location, then you
##' # may need to add a path_cuda_lib argument:
##' dust::dust_cuda_configuration(
##'   path_cuda_lib = "/usr/local/cuda-11.1/lib64",
##'   forget = TRUE, quiet = FALSE)
##'
##' # However, if things are installed in the default location or you
##' # have set the environment variables described above, then this
##' # may work:
##' dust::dust_cuda_configuration(forget = TRUE, quiet = FALSE)
dust_cuda_configuration <- function(path_cuda_lib = NULL,
                                    path_cub_include = NULL,
                                    quiet = TRUE,
                                    forget = FALSE) {
  if (is.null(cache$cuda) || forget) {
    cache$cuda <- cuda_configuration(path_cuda_lib, path_cub_include, quiet)
  }
  cache$cuda
}


##' Create options for compiling for CUDA.  Unless you need to change
##' paths to libraries/headers, or change the debug level you will
##' probably not need to directly use this. However, it's potentially
##' useful to see what is being passed to the compiler.
##'
##' @title Create CUDA options
##'
##' @param ... Arguments passed to [dust::dust_cuda_configuration()]
##'
##' @param debug Logical, indicating if we should compile for debug
##'   (adding `-g`, `-G` and `-O0`)
##'
##' @param profile Logical, indicating if we should enable profiling
##'
##' @param fast_math Logical, indicating if we should enable "fast
##'   maths", which lets the optimiser enable optimisations that break
##'   IEEE compliance and disables some error checking (see [the CUDA
##'   docs](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
##'   for more details).
##'
##' @param flags Optional extra arguments to pass to nvcc. These
##'   options will not be passed to your normal C++ compiler, nor the
##'   linker (for that use R's user Makevars system). This can be used
##'   to do things like tune the maximum number of registers
##'   (`--maxrregcount x`). If not `NULL`, this must be a character
##'   vector, which will be concatenated with spaces between options.
##'
##' @return An object of type `cuda_options`, which can be passed into
##'   [dust::dust] as argument `gpu`
##'
##' @seealso [dust::dust_cuda_configuration] which identifies and
##'   returns the core CUDA configuration (often used implicitly by
##'   this function).
##'
##' @export
##' @examples
##' tryCatch(
##'   dust::dust_cuda_options(),
##'   error = function(e) NULL)
dust_cuda_options <- function(..., debug = FALSE, profile = FALSE,
                              fast_math = FALSE, flags = NULL) {
  info <- dust_cuda_configuration(...)
  if (!info$has_cuda) {
    stop("cuda not supported on this machine")
  }
  cuda_options(info, debug, profile, fast_math, flags)
}


cuda_configuration <- function(path_cuda_lib = NULL, path_cub_include = NULL,
                               quiet = FALSE) {
  no_cuda <- list(
    has_cuda = FALSE,
    cuda_version = NULL,
    devices = NULL,
    path_cuda_lib = NULL,
    path_cub_include = NULL)

  tryCatch({
    path_cuda_lib <- cuda_path_cuda_lib(path_cuda_lib)
    dat <- cuda_create_test_package(path_cuda_lib)
    pkg <- pkgload::load_all(dat$path, export_all = FALSE, quiet = quiet,
                             helpers = FALSE, attach_testthat = FALSE)
    on.exit(pkgload::unload(dat$name))
    info <- pkg$env$dust_gpu_info()
    path_cub_include <-
      cuda_path_cub_include(info$cuda_version, path_cub_include)
    paths <- list(
      path_cuda_lib = path_cuda_lib,
      path_cub_include = path_cub_include)

    c(info, paths)
  }, error = function(e) {
    if (!quiet) {
      message("nvcc detection reported failure:\n", e$message)
    }
    no_cuda
  })
}


cuda_path_cuda_lib <- function(path) {
  check_path <- function(path, reason) {
    if (!any(grepl("^libcudart", dir(path), ignore.case = TRUE))) {
      stop(sprintf("Did not find 'libcudart' within '%s' (via %s)",
                   path, reason))
    }
  }
  if (!is.null(path)) {
    check_path(path, "provided argument")
    return(path)
  }
  path <- Sys.getenv("DUST_PATH_CUDA_LIB", NA_character_)
  if (!is.na(path)) {
    check_path(path, "environment variable 'DUST_PATH_CUDA_LIB'")
    return(path)
  }
  NULL
}


cuda_path_cub_include <- function(version, path) {
  check_path <- function(path, reason) {
    if (!file.exists(file.path(path, "cub"))) {
      stop(sprintf("Did not find directory 'cub' within '%s' (via %s)",
                   path, reason))
    }
  }
  if (!is.null(path)) {
    check_path(path, "provided argument")
    return(path)
  }
  if (version >= numeric_version("11.0.0")) {
    return(NULL)
  }
  path <- Sys.getenv("DUST_PATH_CUB_INCLUDE", NA_character_)
  if (!is.na(path)) {
    check_path(path, "environment variable 'DUST_PATH_CUB_INCLUDE'")
    return(path)
  }
  path <- cuda_cub_path_default()
  if (!is.null(path)) {
    check_path(path, "default path (R >= 4.0.0)")
    return(path)
  }

  stop("Did not find cub headers, see ?dust_cuda_configuration")
}


cuda_cub_path_default <- function(r_version = getRversion()) {
  if (r_version < numeric_version("4.0.0")) {
    return(NULL)
  }
  file.path(tools::R_user_dir("dust", "data"), "cub")
}


cuda_flag_helper <- function(value, prefix) {
  if (is.null(value)) {
    ""
  } else {
    sprintf("%s%s", prefix, value)
  }
}


cuda_create_test_package <- function(path_cuda_lib, path = tempfile()) {
  stopifnot(!file.exists(path))

  suffix <- paste(sample(c(0:9, letters[1:6]), 8, TRUE), collapse = "")
  base <- paste0("dust", suffix)

  path_src <- file.path(path, "src")
  dir.create(path_src, FALSE, TRUE)
  data <- list(base = base,
               path_dust_include = dust_file("include"),
               linking_to = "cpp11",
               compiler_options = "",
               system_requirements = "R (>= 4.0.0)",
               cuda = list(gencode = "",
                           nvcc_flags = "-O0",
                           cub_include = "",
                           lib_flags = cuda_flag_helper(path_cuda_lib, "-L")))

  file.copy(dust_file("cuda/dust.cu"), path_src)
  file.copy(dust_file("cuda/dust.hpp"), path_src)
  substitute_dust_template(data, "Makevars.cuda",
                           file.path(path_src, "Makevars"))
  substitute_dust_template(data, "DESCRIPTION",
                           file.path(path, "DESCRIPTION"))
  substitute_dust_template(data, "NAMESPACE",
                           file.path(path, "NAMESPACE"))
  list(name = base, path = path)
}


cuda_install_cub <- function(path, version = "1.9.10", quiet = FALSE) {
  path <- path %||% cuda_cub_path_default()
  if (file.exists(path)) {
    stop(sprintf("Path already exists: '%s'", path))
  }

  url <- sprintf("https://github.com/nvidia/cub/archive/%s.zip", version)
  tmp_zip <- tempfile(fileext = ".zip")
  tmp_src <- tempfile()
  utils::download.file(url, tmp_zip, mode = "wb", quiet = quiet)

  dir.create(tmp_src, FALSE, TRUE)
  utils::unzip(tmp_zip, exdir = tmp_src)
  base <- sprintf("cub-%s", version)
  stopifnot(file.exists(file.path(tmp_src, base)))

  ## We currently print this message regardless of the value of
  ## 'quiet', which mostly is there to be passed to download.file,
  ## which otherwise prints all sorts of stuff to stdout.
  message(sprintf("Installing cub headers into %s", path))
  dir.create(path, FALSE, TRUE)
  file.copy(file.path(tmp_src, base, "LICENSE.TXT"), path)
  file.copy(file.path(tmp_src, base, "cub"), path, recursive = TRUE)

  path
}


cuda_options <- function(info, debug, profile, fast_math, flags) {
  if (debug) {
    nvcc_flags <- "-g -G -O0"
  } else {
    nvcc_flags <- "-O2"
  }
  if (profile) {
    nvcc_flags <- paste(nvcc_flags, "-pg --generate-line-info",
                        "-DDUST_ENABLE_CUDA_PROFILER")
  }
  if (fast_math) {
    nvcc_flags <- paste(nvcc_flags, "--use_fast_math")
  }
  if (!is.null(flags)) {
    nvcc_flags <- paste(nvcc_flags, paste(flags, collapse = " "))
  }

  versions <- unique(info$devices$version)
  gencode <- paste(
    sprintf("-gencode=arch=compute_%d,code=sm_%d", versions, versions),
    collapse = " ")

  info$flags <- list(
    nvcc_flags = nvcc_flags,
    gencode = gencode,
    cub_include = cuda_flag_helper(info$path_cub_include, "-I"),
    lib_flags = cuda_flag_helper(info$path_cuda_lib, "-L"))

  class(info) <- "cuda_options"
  info
}


cuda_check <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  if (is.logical(x)) {
    if (x) {
      return(dust_cuda_options())
    } else {
      return(NULL)
    }
  }
  assert_is(x, "cuda_options")
  x
}
