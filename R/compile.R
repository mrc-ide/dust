compile_and_load <- function(filename, type, name = type, quiet = FALSE) {
  hash <- hash_file(filename)
  assert_valid_name(name)
  base <- sprintf("%s%s", name, hash)

  if (!(base %in% names(getLoadedDLLs()))) {
    model <- read_lines(filename)
    data <- list(model = model, name = name, type = type, base = base,
                 path_dust_include = dust_file("include"))

    path <- tempfile()
    path_src <- file.path(path, "src")
    dir.create(path_src, FALSE, TRUE)
    substitute_dust_template(data, "DESCRIPTION",
                             file.path(path, "DESCRIPTION"))
    substitute_dust_template(data, "NAMESPACE",
                             file.path(path, "NAMESPACE"))
    substitute_dust_template(data, "interface.cpp",
                             file.path(path, "src", "interface.cpp"))
    substitute_dust_template(data, "Makevars",
                             file.path(path, "src", "Makevars"))

    pkgbuild::compile_dll(path, compile_attributes = TRUE, quiet = quiet)
    dll <- file.path(path_src, paste0(base, .Platform$dynlib.ext))
    dyn.load(dll)
  }

  v <- c("alloc", "run", "reset", "state", "step", "reorder")
  sym <- getNativeSymbolInfo(sprintf("_%s_%s_%s", base, name, v), base)
  names(sym) <- v

  dust(sym$alloc, sym$run, sym$reset, sym$state, sym$step, sym$reorder)
}


substitute_template <- function(data, src, dest) {
  template <- read_lines(src)
  txt <- glue::glue(template, .envir = data, .open = "{{", .close = "}}")
  writeLines(txt, dest)
}


substitute_dust_template <- function(data, src, dest) {
  substitute_template(data, dust_file(file.path("template", src)), dest)
}
