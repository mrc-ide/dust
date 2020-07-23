r6_private <- function(x) {
  environment(x$initialize)$private
}


has_openmp <- function() {
  openmp_info()$has_openmp
}


create_test_package <- function(name = "pkg", path = tempfile()) {
  dir.create(path)
  dir.create(file.path(path, "inst/dust"), FALSE, TRUE)
  dir.create(file.path(path, "R"), FALSE, TRUE)
  dir.create(file.path(path, "src"), FALSE, TRUE)

  data <- list(name = name)
  writeLines(glue_whisker(read_lines("examples/pkg/DESCRIPTION"), data),
             file.path(path, "DESCRIPTION"))
  writeLines(glue_whisker(read_lines("examples/pkg/NAMESPACE"), data),
             file.path(path, "NAMESPACE"))
  file.copy(dust_file("examples/walk.cpp"), file.path(path, "inst/dust"))
  file.copy(dust_file("examples/sir.cpp"), file.path(path, "inst/dust"))

  path
}


skip_if_no_gpu <- function() {
  testthat::skip_on_travis()
  testthat::skip_on_appveyor()
  if (!nzchar(Sys.which("nvcc"))) {
    testthat::skip("No nvcc found")
  }
}
