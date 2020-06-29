context("package")

test_that("validate package", {
  skip_if_not_installed("pkgload")
  path <- tempfile()
  dir.create(path)
  dir.create(file.path(path, "inst/dust"), FALSE, TRUE)

  name <- "pkg"

  data <- list(name = name)
  writeLines(glue_whisker(read_lines("examples/pkg/DESCRIPTION"), data),
             file.path(path, "DESCRIPTION"))
  writeLines(glue_whisker(read_lines("examples/pkg/NAMESPACE"), data),
             file.path(path, "NAMESPACE"))
  file.copy(dust_file("examples/walk.cpp"), file.path(path, "inst/dust"))
  file.copy(dust_file("examples/sir.cpp"), file.path(path, "inst/dust"))

  path <- dust_package(path)

  pkgbuild::compile_dll(path, quiet = TRUE)
  res <- pkgload::load_all(path)
  w <- res$env$walk$new(list(sd = 1), 0, 100)
  expect_equal(w$state(), matrix(0, 1, 100))
  rm(w)
  gc()
  pkgload::unload("pkg")
})
