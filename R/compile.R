generate_dust <- function(filename, quiet, workdir, cuda, linking_to, cpp_std,
                          compiler_options, optimisation_level, skip_cache,
                          mangle) {
  config <- parse_metadata(filename)
  if (grepl("^[A-Za-z][A-Zxa-z0-9]*$", config$name)) {
    base <- config$name
  } else {
    base <- "dust"
  }
  if (mangle) {
    base <- paste0(base, hash_file(filename))
  }
  gpu <- isTRUE(cuda$has_cuda)
  if (gpu) {
    base <- paste0(base, "gpu")
  }

  if (cache$models$has_key(base, skip_cache)) {
    return(cache$models$get(base, skip_cache))
  }

  path <- dust_workdir(workdir)
  model <- read_lines(filename)
  reload <- list(path = path, base = base)
  data <- dust_template_data(model, config, cuda, reload, linking_to, cpp_std,
                             compiler_options, optimisation_level)

  ## This set of options is used in the non-package version only
  data$base <- base
  data$path_dust_include <- dust_file("include")
  data$system_requirements <- data$cpp_std %||% "R (>= 4.0.0)"

  dir.create(file.path(path, "R"), FALSE, TRUE)
  dir.create(file.path(path, "src"), FALSE, TRUE)

  substitute_dust_template(data, "DESCRIPTION",
                           file.path(path, "DESCRIPTION"))
  substitute_dust_template(data, "NAMESPACE",
                           file.path(path, "NAMESPACE"))

  if (is.null(cuda)) {
    cpp_ext <- ".cpp"
    substitute_dust_template(data, "Makevars",
                             file.path(path, "src", "Makevars"))
  } else {
    cpp_ext <- ".cu"
    substitute_dust_template(data, "Makevars.cuda",
                             file.path(path, "src", "Makevars"))
  }

  code <- dust_code(data, config)
  writeLines(code$r, file.path(path, "R/dust.R"))
  writeLines(code$cpp, file.path(path, paste0("src/dust", cpp_ext)))
  writeLines(code$hpp, file.path(path, "src/dust.hpp"))

  res <- list(key = base, gpu = gpu, data = data, path = path)
  cache$models$set(base, res, skip_cache)
  res
}


dust_code <- function(data, config) {
  dust_r <- drop_roxygen(
    substitute_dust_template(data, "dust.R.template", NULL))

  dust_cpp <- c(substitute_dust_template(data, "dust.cpp", NULL),
                substitute_dust_template(data, "dust_methods.cpp", NULL))
  dust_hpp <- c(substitute_dust_template(data, "dust.hpp", NULL),
                substitute_dust_template(data, "dust_methods.hpp", NULL))

  if (config$has_gpu_support) {
    data_gpu <- data
    data_gpu$target <- "gpu"
    data_gpu$container <- "dust_gpu"
    dust_cpp <- c(dust_cpp,
                  substitute_dust_template(data_gpu, "dust_methods.cpp", NULL))
    dust_hpp <- c(dust_hpp,
                  substitute_dust_template(data_gpu, "dust_methods.hpp", NULL))
  }

  ret <- list(r = dust_r,
              hpp = dust_hpp,
              cpp = dust_cpp)

  lapply(ret, drop_internal_comments)
}


compile_and_load <- function(filename, quiet = FALSE, workdir = NULL,
                             cuda = NULL, linking_to = NULL, cpp_std = NULL,
                             compiler_options = NULL, optimisation_level = NULL,
                             skip_cache = FALSE) {
  mangle <- TRUE
  res <- generate_dust(filename, quiet, workdir, cuda, linking_to, cpp_std,
                       compiler_options, optimisation_level, skip_cache, mangle)

  if (is.null(res$env)) {
    path <- res$path

    pkgbuild::compile_dll(path, compile_attributes = TRUE,
                          quiet = quiet, debug = FALSE)
    res$env <- load_temporary_package(path, res$data$base, quiet)
    res$dll <- file.path(path, "src", paste0(res$key, .Platform$dynlib.ext))
    res$gen <- res$env[[res$data$name]]

    cache$models$set(res$key, res, skip_cache)
  } else if (!quiet) {
    message("Using cached model")
  }

  res$gen
}


substitute_template <- function(data, src, dest) {
  template <- read_lines(src)
  txt <- glue_whisker(template, data)
  if (is.null(dest)) {
    return(txt)
  }
  writelines_if_changed(txt, dest)
}


substitute_dust_template <- function(data, src, dest) {
  substitute_template(data, dust_file(file.path("template", src)), dest)
}


glue_whisker <- function(template, data) {
  stopifnot(length(template) == 1L)
  glue::glue_data(data, template, .open = "{{", .close = "}}", .trim = FALSE)
}


dust_template_data <- function(model, config, cuda, reload_data, linking_to,
                               cpp_std, compiler_options, optimisation_level) {
  methods <- function(target) {
    nms <- c("alloc", "run", "simulate", "run_adjoint", "set_index", "n_state",
             "update_state", "state", "time", "reorder", "resample",
             "rng_state", "set_rng_state", "set_n_threads",
             "set_data", "compare_data", "filter", "set_stochastic_schedule",
             "ode_statistics")
    m <- sprintf("%s = dust_%s_%s_%s", nms, target, config$name, nms)
    sprintf("list(\n%s)",  paste("          ", m, collapse = ",\n"))
  }
  target <- if (config$time_type == "discrete") "cpu" else "ode"
  container <- sprintf("dust_%s", target)
  methods_cpu <- methods(target)

  if (config$has_gpu_support) {
    ## TODO: make sure in the config nobody tries for ode + gpu, that
    ## won't work; don't error here, but earlier.
    stopifnot(config$time_type == "discrete")
    methods_gpu <- methods("gpu")
  } else {
    methods_gpu <- paste(
      "list(alloc = function(...) {",
      '          stop("GPU support not enabled for this object")',
      "        })", sep = "\n")
  }


  if (is.null(reload_data)) {
    reload <- "NULL"
  } else {
    reload <- paste(deparse(reload_data), collapse = "\n")
  }

  if (!is.null(linking_to)) {
    assert_is(linking_to, "character")
  }
  linking_to <- paste(union("cpp11", linking_to), collapse = ", ")

  cpp_std <- validate_cpp_std(cpp_std)
  compiler_options <- build_compiler_options(compiler_options,
                                             optimisation_level)

  list(model = model,
       name = config$name,
       class = config$class,
       param = deparse_param(config$param),
       cuda = cuda$flags,
       target = target,
       container = container,
       has_gpu_support = as.character(config$has_gpu_support),
       methods_cpu = methods_cpu,
       methods_gpu = methods_gpu,
       reload = reload,
       linking_to = linking_to,
       cpp_std = cpp_std,
       compiler_options = compiler_options)
}


load_temporary_package <- function(path, base, quiet) {
  pkg <- pkgload::load_all(path, compile = FALSE, recompile = FALSE,
                           warn_conflicts = FALSE, export_all = FALSE,
                           helpers = FALSE, attach_testthat = FALSE,
                           quiet = quiet)
  detach(paste0("package:", base), character.only = TRUE)
  pkg$env
}


##' Repair the environment of a dust object created by [[dust::dust]]
##' and then saved and reloaded by [[saveRDS]] and
##' [[readRDS]]. Because we use a fake temporary package to hold the
##' generated code, it will not ordinarily be loaded properly without
##' using this.
##'
##' @title Repair dust environment
##'
##' @param generator The dust generator
##'
##' @param quiet Logical, indicating if we should be quiet (default
##'   prints some progress information)
##'
##' @return Nothing, called for its side effects
##' @export
dust_repair_environment <- function(generator, quiet = FALSE) {
  assert_is(generator, "dust_generator")
  data <- generator$private_fields$reload_
  if (is.null(data)) {
    if (!quiet) {
      message("Generator does not need repair")
    }
    return(invisible())
  }

  base <- data$base
  path <- data$path
  if (!pkgload::is_dev_package(base)) {
    env <- load_temporary_package(path, base, quiet)
  } else {
    if (!quiet) {
      message(sprintf("'%s' was already loaded", base))
    }
    env <- .getNamespace(base)
  }
  if (!identical(env, generator$parent_env)) {
    generator$parent_env <- env
    for (nm in names(generator$public_methods)) {
      environment(generator$public_methods[[nm]]) <- env
    }
  }
}


is_valid_cpp_std <- function(cpp_std) {
  grepl("^C\\+\\+[0-9][0-9a-z]$", cpp_std, ignore.case = TRUE)
}


validate_cpp_std <- function(cpp_std) {
  if (is.null(cpp_std)) {
    return(NULL)
  }
  assert_is(cpp_std, "character")
  if (length(cpp_std) != 1L) {
    stop("Expected a scalar character for 'cpp_std'")
  }
  if (!is_valid_cpp_std(cpp_std)) {
    stop("'cpp_std' does not look like a valid C++ standard name (e.g., C++14)")
  }
  cpp_std
}


build_compiler_options <- function(compiler_options, optimisation_level) {
  if (!is.null(optimisation_level)) {
    opts <- switch(
      optimisation_level,
      none = "-O0",
      standard = "-O2",
      max = c("-O3", "-ffast-math"),
      stop(sprintf("Unknown optimisation_level '%s'", optimisation_level)))
    compiler_options <- c(compiler_options, opts)
  }
  paste(compiler_options, collapse = " ")
}
