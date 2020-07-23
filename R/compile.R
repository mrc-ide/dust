generate_dust <- function(filename, base, type, name, gpu, workdir = NULL) {
  model <- read_lines(filename)
  data <- list(model = model, name = name, type = type, base = base,
               path_dust_include = dust_file("include"))

  path <- dust_workdir(workdir)
  dir.create(file.path(path, "src"), FALSE, TRUE)
  substitute_dust_template(data, "DESCRIPTION",
                           file.path(path, "DESCRIPTION"))
  substitute_dust_template(data, "NAMESPACE",
                           file.path(path, "NAMESPACE"))
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

  data$path <- path
  data
}


compile_and_load <- function(filename, type, name, quiet = FALSE,
                             workdir = NULL, gpu = FALSE) {
  hash <- hash_file(filename)
  assert_valid_name(name)
  if (gpu) {
    name <- paste0(name, "_gpu")
  }
  base <- sprintf("%s%s", name, hash)

  if (!base %in% names(cache)) {
    data <- generate_dust(filename, base, type, name, gpu, workdir)

    cpp11::cpp_register(data$path, quiet = quiet)
    compile_dll(data$path, compile_attributes = TRUE, quiet = quiet)
    dll <- file.path(data$path, "src", paste0(base, .Platform$dynlib.ext))
    dyn.load(dll)

    template_r <- read_lines(dust_file("template/dust.R.template"))
    writeLines(glue_whisker(template_r, data),
               file.path(data$path, "R", "dust.R"))

    env <- new.env(parent = topenv())
    for (f in dir(file.path(data$path, "R"), full.names = TRUE)) {
      sys.source(f, env)
    }

    cache[[base]] <- env[[name]]
  }

  cache[[base]]
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
