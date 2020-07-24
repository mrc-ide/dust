generate_dust <- function(filename, type, name, gpu, quiet, workdir, cache) {
  hash <- hash_file(filename)
  if (gpu) {
    name <- paste0(name, "gpu")
  }
  base <- sprintf("%s%s", name, hash)

  if (base %in% names(cache)) {
    return(cache[[base]])
  }

  assert_valid_name(name)
  model <- read_lines(filename)
  data <- list(model = model, name = name, type = type, base = base,
               path_dust_include = dust_file("include"))

  path <- dust_workdir(workdir)
  dir.create(file.path(path, "R"), FALSE, TRUE)
  dir.create(file.path(path, "src"), FALSE, TRUE)

  substitute_dust_template(data, "DESCRIPTION",
                           file.path(path, "DESCRIPTION"))
  substitute_dust_template(data, "NAMESPACE",
                           file.path(path, "NAMESPACE"))
  substitute_dust_template(data, "dust.R.template",
                           file.path(path, "R/dust.R"))
  if (gpu) {
    substitute_dust_template(data, "gpu/dust.cu",
                             file.path(path, "src", "dust.cu"))
    substitute_dust_template(data, "gpu/dust.hpp",
                             file.path(path, "src", "dust.hpp"))
    substitute_dust_template(data, "gpu/Makevars",
                             file.path(path, "src", "Makevars"))
  } else {
    substitute_dust_template(data, "dust.cpp",
                             file.path(path, "src", "dust.cpp"))
    substitute_dust_template(data, "Makevars",
                             file.path(path, "src", "Makevars"))
  }

  cpp11::cpp_register(path, quiet = quiet)

  res <- list(key = base, data = data, path = path)
  cache[[res$key]] <- res
  res
}


compile_and_load <- function(filename, type, name, quiet = FALSE,
                             workdir = NULL, gpu = FALSE) {
  res <- generate_dust(filename, type, name, gpu, quiet, workdir, cache)

  if (is.null(res$env)) {
    path <- res$path

    compile_dll(path, compile_attributes = TRUE, quiet = quiet)
    dll <- file.path(path, "src", paste0(res$key, .Platform$dynlib.ext))
    dyn.load(dll)

    env <- new.env(parent = topenv())
    for (f in dir(file.path(path, "R"), full.names = TRUE)) {
      sys.source(f, env)
    }

    res$dll <- dll
    res$env <- env
    res$gen <- env[[res$data$name]]
    cache[[res$key]] <- res
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


## This is a workaround for pkgbuild wanting to build
## debug/unoptimised dlls by default, unless the user has provided a
## Makevars
has_user_makevars <- function() {
  length(environment(pkgbuild::compile_dll)$makevars_user()) > 0
}


compile_dll <- function(...) {
  if (has_user_makevars()) {
    pkgbuild::compile_dll(...)
  } else {
    makevars <- tempfile()
    file.create(makevars)
    on.exit(unlink(makevars))
    withr::with_envvar(
      c("R_MAKEVARS_USER" = makevars),
      pkgbuild::compile_dll(...))
  }
}
