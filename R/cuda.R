##' Detect CUDA configuration. This is experimental and subject to
##' change. This function tries to compile a small program with `nvcc`
##' and confirms that this can be loaded into R, then uses that
##' program to query the presence and capabilities of your NVIDIA
##' GPUs. If this works, then you can use the GPU-enabled dust
##' features, and the infomation returned will help us.  It's quite
##' slow to execute (several seconds) so we cache the value within a
##' session.  Later versions of dust will cache this across sessions
##' too.
##'
##' If you are using older CUDA (< 11.0.0) then you need to provide
##' [CUB](https://nvlabs.github.io/cub/) headers, which we use to
##' manage state on the device (these are included in CUDA 11.0.0 and
##' higher). You can provide this as:
##'
##' * a path to this function (`path_cub_include`)
##' * the environment variable DUST_CUB_PATH_INCLUDE
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
##' older versions of R and set this path as DUST_CUB_PATH_INCLUDE).
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
##' @export
##' @examples
##' dust::dust_cuda_configuration()
dust_cuda_configuration <- function(path_cuda_lib = NULL,
                                    path_cub_include = NULL,
                                    quiet = TRUE,
                                    forget = FALSE) {
  no_cuda <- list(
    has_cuda = NULL,
    cuda_version = NULL,
    devices = NULL,
    path_cuda_lib = NULL,
    path_cub_include = NULL)
  tryCatch(
    cuda_configuration(path_cuda_lib, path_cub_include, quiet, forget),
    error = function(e) no_cuda)
}


cuda_configuration <- function(path_cuda_lib = NULL, path_cub_include = NULL,
                               quiet = FALSE, forget = FALSE) {
  if (is.null(cache$cuda) || forget) {
    dat <- cuda_create_test_package(path_cuda_lib)
    pkg <- pkgload::load_all(dat$path, export_all = FALSE, quiet = quiet,
                             helpers = FALSE, attach_testthat = FALSE)
    on.exit(pkgload::unload(dat$name))

    info <- pkg$env$dust_device_info()
    path_cub_include <-
      cuda_path_cub_include(info$cuda_version, path_cub_include)
    paths <- list(
      path_cuda_lib = path_cuda_lib,
      path_cub_include = path_cub_include)
    cache$cuda <- c(info, paths)
  }

  cache$cuda
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
  path <- Sys.getenv("DUST_CUB_PATH_INCLUDE", NA_character_)
  if (!is.na(path)) {
    check_path(path, "environment variable 'DUST_CUB_PATH_INCLUDE'")
    return(path)
  }
  path <- cuda_cub_path_default()
  if (file.exists(file.path(path, "cub"))) {
    check_path(path, "default path (R >= 4.0.0)")
    return(path)
  }

  stop("Did not find cub sources, please install and set DUST_CUB_PATH_INCLUDE")
}


cuda_cub_path_default <- function(r_version = getRversion()) {
  if (r_version < numeric_version("4.0.0")) {
    return(NULL)
  }
  file.path(tools::R_user_dir("dust", "data"), "cub")
}


cuda_lib_flags <- function(path_cuda_lib) {
  if (is.null(path_cuda_lib)) {
    ""
  } else {
    sprintf("-L%s", path_cuda_lib)
  }
}


cuda_create_test_package <- function(path_cuda_lib = NULL, path = tempfile()) {
  stopifnot(!file.exists(path))

  suffix <- paste(sample(c(0:9, letters[1:6]), 8, TRUE), collapse = "")
  base <- paste0("dust", suffix)

  path_src <- file.path(path, "src")
  dir.create(path_src, FALSE, TRUE)
  data <- list(base = base,
               path_dust_include = dust_file("include"),
               cuda_lib_flags = cuda_lib_flags(path_cuda_lib))
  file.copy(dust_file("cuda/device_info.cu"), path_src)
  file.copy(dust_file("cuda/device_info.hpp"), path_src)
  substitute_template(data, dust_file("cuda/Makevars"),
                      file.path(path_src, "Makevars"))
  substitute_dust_template(data, "DESCRIPTION",
                           file.path(path, "DESCRIPTION"))
  substitute_dust_template(data, "NAMESPACE",
                           file.path(path, "NAMESPACE"))
  list(name = base, path = path)
}


cuda_version <- function(x) {
  numeric_version(paste(x, collapse = "."))
}


cuda_install_cub <- function(path, version = "1.9.10") {
  if (is.null(path)) {
    stopifnot(getRversion() >= "4.0.0")
    path <- cuda_cub_path_default()
  }
  if (file.exists(path)) {
    stop(sprintf("Path already exists: '%s'", path))
  }

  url <- sprintf("https://github.com/nvidia/cub/archive/%s.zip", version)
  tmp_zip <- tempfile(fileext = ".zip")
  tmp_src <- tempfile()
  download.file(url, tmp_zip, mode = "wb")

  dir.create(tmp_src, FALSE, TRUE)
  unzip(tmp_zip, exdir = tmp_src)
  base <- sprintf("cub-%s", version)
  stopifnot(file.exists(file.path(tmp_src, base)))

  message(sprintf("Installing cub headers into %s", path))
  dir.create(path, FALSE, TRUE)
  file.copy(file.path(tmp_src, base, "LICENSE.TXT"), path)
  file.copy(file.path(tmp_src, base, "cub"), path, recursive = TRUE)

  path
}
