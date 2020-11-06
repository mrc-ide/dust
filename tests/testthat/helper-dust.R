r6_private <- function(x) {
  environment(x$initialize)$private
}


create_test_package <- function(name = "pkg", path = tempfile(),
                                examples = c("walk.cpp", "sir.cpp")) {
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
