compile_and_load <- function(filename, type, name, quiet = FALSE,
                             workdir = NULL) {
  hash <- hash_file(filename)
  assert_valid_name(name)
  base <- sprintf("%s%s", name, hash)

  if (!(base %in% names(getLoadedDLLs()))) {
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

    pkgbuild::compile_dll(path, compile_attributes = TRUE, quiet = quiet)
    dll <- file.path(path, "src", paste0(base, .Platform$dynlib.ext))
    dyn.load(dll)
  }

  v <- c("alloc", "run", "reset", "state", "step", "reorder")
  sym <- getNativeSymbolInfo(sprintf("_%s_dust_%s_%s", base, name, v), base)
  names(sym) <- v

  dust_class(sym$alloc, sym$run, sym$reset, sym$state, sym$step, sym$reorder)
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
  glue::glue(template, .envir = data, .open = "{{", .close = "}}")
}
