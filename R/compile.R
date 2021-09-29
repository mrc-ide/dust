generate_dust <- function(filename, quiet, workdir, cuda, skip_cache, mangle) {
  config <- parse_metadata(filename)
  if (mangle) {
    base <- sprintf("%s%s", config$name, hash_file(filename))
  } else {
    base <- config$name
  }
  gpu <- isTRUE(cuda$has_cuda)
  if (gpu) {
    base <- paste0(base, "gpu")
  }

  if (cache$models$has_key(base, skip_cache)) {
    return(cache$models$get(base, skip_cache))
  }

  model <- read_lines(filename)
  data <- dust_template_data(model, config, cuda)

  ## These two are used in the non-package version only
  data$base <- base
  data$path_dust_include <- dust_file("include")

  path <- dust_workdir(workdir)
  dir.create(file.path(path, "R"), FALSE, TRUE)
  dir.create(file.path(path, "src"), FALSE, TRUE)

  substitute_dust_template(data, "DESCRIPTION",
                           file.path(path, "DESCRIPTION"))
  substitute_dust_template(data, "NAMESPACE",
                           file.path(path, "NAMESPACE"))
  substitute_dust_template(data, "dust.R.template",
                           file.path(path, "R/dust.R"))
  substitute_dust_template(data, "dust.hpp",
                           file.path(path, "src", "dust.hpp"))

  if (is.null(cuda)) {
    substitute_dust_template(data, "dust.cpp",
                             file.path(path, "src", "dust.cpp"))
    substitute_dust_template(data, "Makevars",
                             file.path(path, "src", "Makevars"))
  } else {
    substitute_dust_template(data, "dust.cpp",
                             file.path(path, "src", "dust.cu"))
    substitute_dust_template(data, "Makevars.cuda",
                             file.path(path, "src", "Makevars"))
  }

  ## Keep the generated dust files simple by dropping roxygen docs
  ## which are used in making the interface docs (?dust_generator) and
  ## internal comments which remind developers about next steps after
  ## modifying files.
  dust_r <- drop_internal_comments(readLines(file.path(path, "R/dust.R")))
  writeLines(drop_roxygen(dust_r), file.path(path, "R/dust.R"))

  dust_cpp <- drop_internal_comments(readLines(file.path(path, "src/dust.cpp")))
  writeLines(dust_cpp, file.path(path, "src/dust.cpp"))

  dust_hpp <- drop_internal_comments(readLines(file.path(path, "src/dust.hpp")))
  writeLines(dust_hpp, file.path(path, "src/dust.hpp"))

  res <- list(key = base, gpu = gpu, data = data, path = path)
  cache$models$set(base, res, skip_cache)
  res
}


compile_and_load <- function(filename, quiet = FALSE, workdir = NULL,
                             cuda = NULL, skip_cache = FALSE) {
  res <- generate_dust(filename, quiet, workdir, cuda, skip_cache, TRUE)

  if (is.null(res$env)) {
    path <- res$path

    pkgbuild::compile_dll(path, compile_attributes = TRUE,
                          quiet = quiet, debug = FALSE)
    tmp <- pkgload::load_all(path, compile = FALSE, recompile = FALSE,
                             warn_conflicts = FALSE, export_all = FALSE,
                             helpers = FALSE, attach_testthat = FALSE,
                             quiet = quiet)
    ## Don't pollute the search path
    detach(paste0("package:", res$data$base), character.only = TRUE)

    res$dll <- file.path(path, "src", paste0(res$key, .Platform$dynlib.ext))
    res$env <- tmp$env
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
  writelines_if_changed(txt, dest)
}


substitute_dust_template <- function(data, src, dest) {
  substitute_template(data, dust_file(file.path("template", src)), dest)
}


glue_whisker <- function(template, data) {
  if (length(template) > 1L) {
    template <- paste(template, collapse = "\n")
  }
  glue::glue(template, .envir = data, .open = "{{", .close = "}}",
             .trim = FALSE)
}


dust_template_data <- function(model, config, cuda) {
  list(model = model,
       name = config$name,
       class = config$class,
       param = deparse_param(config$param),
       cuda = cuda$flags)
}
