context("package")

test_that("validate package", {
  skip_if_not_installed("pkgload")

  path <- create_test_package()
  file.remove(file.path(path, "src")) # ensure we create this as needed

  path <- dust_package(path)

  pkgbuild::compile_dll(path, quiet = TRUE)
  res <- pkgload::load_all(path)
  w <- res$env$walk$new(list(sd = 1), 0, 100)
  expect_equal(w$state(), matrix(0, 1, 100))
  rm(w)
  gc()
  pkgload::unload("pkg")
})


test_that("validate package dependencies", {
  deps <- data.frame(
    type = c("LinkingTo", "LinkingTo", "Imports"),
    package = c("Rcpp", "dust", "Rcpp"),
    version = "*",
    stringsAsFactors = FALSE)
  expect_silent(package_validate_has_dep(deps, "Rcpp", "LinkingTo"))
  expect_silent(package_validate_has_dep(deps, "Rcpp", "Imports"))
  expect_error(
    package_validate_has_dep(deps, "Rcpp", "Depends"),
    "Expected package 'Rcpp' as 'Depends' in DESCRIPTION")
  expect_error(
    package_validate_has_dep(deps, "other", "Imports"),
    "Expected package 'other' as 'Imports' in DESCRIPTION")
})


test_that("validate destination notices existing C++ code", {
  msg <- "File '.+\\.cpp' does not look like it was created by dust - stopping"
  path <- create_test_package()
  file.create(file.path(path, "src", "walk.cpp"))
  expect_error(
    package_validate_destination(path, c("sir.cpp", "walk.cpp")),
    msg)

  writeLines("// some actual content", file.path(path, "src", "walk.cpp"))
  expect_error(
    package_validate_destination(path, c("sir.cpp", "walk.cpp")),
    msg)
})
