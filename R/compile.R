## Ideally we'll move to using pkgbuild here as it will do a better
## job of rigging up a sensible build system, but that does not
## support standalone shlibs out the box - this can be swapped out
## easily enough later though.
compile <- function(filename, preclean = FALSE) {
  Sys.setenv(R_TESTS = "")
  path <- dirname(filename)
  owd <- setwd(path)
  on.exit(setwd(owd))
  src <- basename(filename)
  base <- tools::file_path_sans_ext(src)
  dll <- paste0(base, .Platform$dynlib.ext)
  if (file.exists(dll)) {
    message("Using previously compiled shared library")
  } else {
    message("Compiling shared library")
    r_bin <- file.path(R.home(), "bin", "R")
    args <- c("CMD", "SHLIB", src, "-o", dll,
              if (preclean) c("--preclean", "--clean"))
    output <- suppressWarnings(system2(r_bin, args,
                                       stdout = TRUE, stderr = TRUE))
    ok <- attr(output, "status")
    error <- !is.null(ok) && ok != 0L
    if (error) {
      message(paste(output, collapse = "\n"))
      stop("Error compiling source")
    }
  }
  list(path = path, base = base, src = normalize_path(src),
       dll = normalize_path(dll))
}


compile_and_load <- function(filename, type, name = type) {
  path <- dust_build_path()

  model <- read_lines(filename)
  hash <- hash_file(filename)
  data <- list(model = model, name = name, type = type)

  template <- read_lines(dust_file("template/interface.cpp"))
  txt <- glue::glue(template, .envir = data, .open = "{{", .close = "}}")
  dest <- sprintf("%s/%s_%s.cpp", path, data$name, hash)
  writeLines(txt, dest)

  res <- compile(dest)

  if (!(res$dll %in% names(getLoadedDLLs()))) {
    dyn.load(res$dll)
  }

  v <- c("alloc", "run", "reset", "state", "step")
  sym <- getNativeSymbolInfo(sprintf("%s_%s", data$name, v), res$base)
  names(sym) <- v

  dust(sym$alloc, sym$run, sym$reset, sym$state, sym$step)
}


dust_build_path <- function() {
  if (is.null(cache$workdir)) {
    cache$workdir <- tempfile()
  }
  if (!file.exists(file.path(cache$workdir, "Makevars"))) {
    dir.create(cache$workdir, FALSE, TRUE)
    txt <- read_lines(dust_file("template/Makevars"))
    writeLines(
      sub("$(PATH_DUST_INCLUDE)", dust_file("include"), txt, fixed = TRUE),
      file.path(cache$workdir, "Makevars"))
  }
  normalize_path(cache$workdir)
}
