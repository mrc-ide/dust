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
    list(filename, "type", "name", TRUE, workdir, FALSE))
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
  dir.create(file.path(p, "R"))
  files <- c("DESCRIPTION", "NAMESPACE",
             "src/Makevars", "src/dust.cpp", "src/cpp11.cpp",
             "R/dust.R", "R/cpp11.R")
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


test_that("validate interface", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "mywalk",
                          quiet = TRUE)
  cmp <- dust_class

  expect_setequal(names(res$public_methods),
                  names(cmp$public_methods))
  for (m in names(res$public_methods)) {
    expect_identical(formals(res$public_methods[[m]]),
                     formals(cmp$public_methods[[m]]))
  }
})


test_that("validate package interface", {
  tmp <- tempfile(fileext = ".R")
  template <- read_lines(dust_file("template/dust.R.template"))
  writeLines(glue_whisker(template, list(name = "testing")), tmp)
  env <- new.env()
  sys.source(tmp, env)
  res <- env$testing
  cmp <- dust_class

  expect_setequal(names(res$public_methods),
                  names(cmp$public_methods))
  for (m in names(res$public_methods)) {
    expect_identical(formals(res$public_methods[[m]]),
                     formals(cmp$public_methods[[m]]))
  }
})


test_that("get rng state", {
  seed <- 1
  np <- 10
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "mywalk",
                          quiet = TRUE)
  obj <- res$new(list(sd = 1), 0L, np, seed = seed)
  expect_identical(
    obj$rng_state(),
    dust_rng$new(seed, np)$state())
})


test_that("setting a named index returns names", {
  res <- dust(dust_file("examples/sirs.cpp"), quiet = TRUE)
  mod <- res$new(list(), 0, 10)

  mod$set_index(3:1)
  expect_identical(
    mod$run(0),
    rbind(rep(0, 10), rep(10, 10), rep(1000, 10)))

  mod$set_index(c(S = 1L, I = 2L, R = 3L))
  expect_identical(
    mod$run(0),
    rbind(S = rep(1000, 10), I = rep(10, 10), R = rep(0, 10)))

  mod$set_index(seq_len(3))
  expect_identical(
    mod$run(0),
    rbind(rep(1000, 10), rep(10, 10), rep(0, 10)))

})


test_that("resetting removes index names", {
  res <- dust(dust_file("examples/variable.cpp"), quiet = TRUE)
  mod <- res$new(list(len = 10), 0, 5)

  mod$set_index(setNames(1:3, c("x", "y", "z")))
  expect_equal(
    mod$run(0),
    matrix(1:3, 3, 5, dimnames = list(c("x", "y", "z"), NULL)))

  mod$reset(list(len = 2), 0)
  expect_equal(
    mod$run(0),
    matrix(1:2, 2, 5))
})


test_that("names are copied when using state()", {
  res <- dust(dust_file("examples/variable.cpp"), quiet = TRUE)
  mod <- res$new(list(len = 10), 0, 5)
  expect_equal(
    mod$state(4:5),
    matrix(4:5, 2, 5))
  expect_equal(
    mod$state(c(x = 4L, y = 5L)),
    matrix(4:5, 2, 5, dimnames = list(c("x", "y"), NULL)))
})
