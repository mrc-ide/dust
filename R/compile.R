generate_dust <- function(filename, quiet, workdir, cuda, cache) {
  config <- parse_metadata(filename)
  hash <- hash_file(filename)
  base <- sprintf("%s%s", config$name, hash)
  gpu <- !is.null(cuda)
  if (gpu) {
    base <- paste0(base, "gpu")
  }

  if (base %in% names(cache$models)) {
    return(cache$models[[base]])
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

  cpp11::cpp_register(path, quiet = quiet)

  res <- list(key = base, gpu = gpu, data = data, path = path)
  cache$models[[base]] <- res
  res
}


compile_and_load <- function(filename, quiet = FALSE, workdir = NULL,
                             cuda = NULL) {
  res <- generate_dust(filename, quiet, workdir, cuda, cache)

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

    cache$models[[res$key]] <- res
  }

  res$gen
}


substitute_template <- function(data, src, dest) {
  template <- read_lines(src)
  txt <- glue_whisker(template, data)
  writeLines(txt, dest)
}


substitute_dust_template <- function(data, src, dest) {
  substitute_template(data, dust_file(file.path("template", src)), dest)
}


glue_whisker <- function(template, data) {
  glue::glue(template, .envir = data, .open = "{{", .close = "}}",
             .trim = FALSE)
}


dust_template_data <- function(model, config, cuda) {
  ret <- list(model = model,
              name = config$name,
              class = config$class,
              param = deparse_param(config$param),
              cuda = cuda)
}
