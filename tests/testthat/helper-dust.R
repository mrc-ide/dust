skip_for_compilation <- function() {
  testthat::skip_on_cran()
}


r6_private <- function(x) {
  environment(x$initialize)$private
}


create_test_package <- function(name = "pkg", path = tempfile(),
                                examples = c("walk.cpp", "sirs.cpp")) {
  dir.create(path, FALSE, TRUE)
  dir.create(file.path(path, "inst/dust"), FALSE, TRUE)
  dir.create(file.path(path, "R"), FALSE, TRUE)
  dir.create(file.path(path, "src"), FALSE, TRUE)

  data <- list(name = name)
  writeLines(glue_whisker(read_lines("examples/pkg/DESCRIPTION"), data),
             file.path(path, "DESCRIPTION"))
  writeLines(glue_whisker(read_lines("examples/pkg/NAMESPACE"), data),
             file.path(path, "NAMESPACE"))
  file.copy(dust_file(file.path("examples", examples)),
            file.path(path, "inst/dust"))

  path
}


helper_metadata <- function(..., base = NULL) {
  code <- readLines(base %||% dust_file("examples/walk.cpp"))
  tmp <- tempfile(fileext = ".cpp")
  writeLines(c(..., code), tmp)
  tmp
}


## This is the systematic resample algorithm as used by mcstate. We
## include it here to confirm that the version in
## inst/include/dust/tools.hpp is correct, as it's surprisingly fiddly.
resample_index <- function(w, u) {
  n <- length(w)
  uu <- u / n + seq(0, by = 1 / n, length.out = n)
  cw <- cumsum(w / sum(w))
  findInterval(uu, cw) + 1L
}


copy_directory <- function(src, as) {
  files <- dir(src, all.files = TRUE, no.. = TRUE, full.names = TRUE)
  dir.create(as, FALSE, TRUE)
  ok <- file.copy(files, as, recursive = TRUE)
  if (!all(ok)) {
    stop("Error copying files")
  }
}
