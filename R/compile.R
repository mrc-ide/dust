compile_and_load <- function(filename, type, name, quiet = FALSE,
                             workdir = NULL) {
  hash <- hash_file(filename)
  assert_valid_name(name)
  base <- sprintf("%s%s", name, hash)

  if (!base %in% names(cache)) {
    model <- read_lines(filename)
    data <- list(model = model, name = name, type = type, base = base,
                 path_dust_include = dust_file("include"))

    path <- dust_workdir(workdir)
    dir.create(file.path(path, "src"), FALSE, TRUE)
    substitute_dust_template(data, "DESCRIPTION",
                             file.path(path, "DESCRIPTION"))
    substitute_dust_template(data, "NAMESPACE",
                             file.path(path, "NAMESPACE"))
    substitute_dust_template(data, "dust.cpp",
                             file.path(path, "src", "dust.cpp"))
    substitute_dust_template(data, "Makevars",
                             file.path(path, "src", "Makevars"))

    cpp11::cpp_register(path)
    pkgbuild::compile_dll(path, compile_attributes = TRUE, quiet = quiet)
    dll <- file.path(path, "src", paste0(base, .Platform$dynlib.ext))
    dyn.load(dll)

    template_r <- read_lines(dust_file("template/dust.R.template"))
    writeLines(glue_whisker(template_r, data),
               file.path(path, "R", "dust.R"))

    env <- new.env(parent = topenv())
    for (f in dir(file.path(path, "R"), full.names = TRUE)) {
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
