context("interface")

test_that("Basic interface use", {
  filename <- dust_file("examples/walk.cpp")
  res <- dust(filename, quiet = TRUE)
  expect_is(res, "R6ClassGenerator")
  obj <- res$new(list(sd = 1), 0L, 100L)
  expect_is(obj, "dust")
})


test_that("Interface passes arguments as expected", {
  skip_if_not_installed("mockery")
  filename <- dust_file("examples/walk.cpp")
  mock_compile_and_load <- mockery::mock(NULL)
  workdir <- tempfile()
  with_mock(
    "dust::compile_and_load" = mock_compile_and_load,
    dust(filename, "type", "name", TRUE, workdir))

  mockery::expect_called(mock_compile_and_load, 1L)
  expect_equal(
    mockery::mock_args(mock_compile_and_load)[[1]],
    list(filename, "type", "name", TRUE, workdir))
})


test_that("guess class", {
  txt <- c("// A comment", "class  whatever {", "};")
  expect_equal(dust_guess_type(txt), "whatever")
  expect_error(dust_guess_type(txt[-2]),
               "Could not automatically detect class name")
  expect_error(dust_guess_type(rep(txt, 2)),
               "Could not automatically detect class name")

  ## Slightly harder cases
  expect_equal(dust_guess_type("\tclass\tTheClass  "), "TheClass")
  expect_equal(dust_guess_type("class TheClass{"), "TheClass")
})


test_that("dust_workdir uses tempdir() if NULL", {
  p <- dust_workdir(NULL)
  expect_equal(normalizePath(dirname(p)), normalizePath(tempdir()))
  expect_false(file.exists(p))
})


test_that("dust_workdir passes nonexistant directories", {
  p <- tempfile()
  expect_equal(dust_workdir(p), p)
  expect_false(file.exists(p))
})


test_that("dust_workdir allows existing empty directories", {
  p <- tempfile()
  dir.create(p, FALSE, TRUE)
  expect_equal(dust_workdir(p), p)
  expect_true(file.exists(p))
  expect_equal(dir(p), character(0))
})


test_that("dust_workdir allows existing dusty directories", {
  p <- tempfile()
  dir.create(p, FALSE, TRUE)
  dir.create(file.path(p, "src"))
  files <- c("DESCRIPTION", "NAMESPACE", "src/Makevars", "src/interface.cpp")
  for (f in files) {
    file.create(file.path(p, f))
  }
  expect_equal(dust_workdir(p), p)

  ## Also allow compilation artefacts
  file.create(file.path(p, "src/interface.o"))
  file.create(file.path(p, "src/interface.dll"))
  file.create(file.path(p, "src/interface.so"))
  expect_equal(dust_workdir(p), p)
})


test_that("dust_workdir will error in directory with extra files", {
  p <- tempfile()
  dir.create(p, FALSE, TRUE)
  dir.create(file.path(p, "src"))
  files <- c("DESCRIPTION", "NAMESPACE", "src/Makevars", "README.md")
  for (f in files) {
    file.create(file.path(p, f))
  }
  expect_error(
    dust_workdir(p),
    "Path '.+' does not look like a dust directory")
})


test_that("dust_workdir will error if path is not a directory", {
  p <- tempfile()
  file.create(p)
  expect_error(
    dust_workdir(p),
    "Path '.+' already exists but is not a directory")
})
