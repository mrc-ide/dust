context("utils")

test_that("null-or-value works", {
  expect_equal(1 %||% NULL, 1)
  expect_equal(1 %||% 2, 1)
  expect_equal(NULL %||% NULL, NULL)
  expect_equal(NULL %||% 2, 2)
})


test_that("valid name", {
  x <- "1name"
  expect_error(assert_valid_name(x),
               "'x' must contain only letters and numbers")
  expect_error(assert_valid_name("foo_bar"),
               "'.+' must contain only letters and numbers")
  expect_error(assert_valid_name("foo.bar"),
               "'.+' must contain only letters and numbers")
})


test_that("assert_file_exists", {
  p <- tempfile()
  expect_error(assert_file_exists(p), "File '.+' does not exist")
  file.create(p)
  expect_silent(assert_file_exists(p))
})


test_that("openmp_info returns environment variables", {
  skip_if_not_installed("withr")
  withr::with_envvar(
    c("OMP_THREAD_LIMIT" = "64", "OMP_NUM_THREADS" = "2"), {
      info <- openmp_info()
      expect_identical(info[["OMP_THREAD_LIMIT"]], 64L)
      expect_identical(info[["OMP_NUM_THREADS"]], 2L)
    })

  withr::with_envvar(
    c("OMP_THREAD_LIMIT" = NA, "OMP_NUM_THREADS" = NA), {
      info <- openmp_info()
      expect_identical(info[["OMP_THREAD_LIMIT"]], NA_integer_)
      expect_identical(info[["OMP_NUM_THREADS"]], NA_integer_)
    })
})


test_that("assert_is", {
  thing <- structure(1, class = c("a", "b"))
  expect_silent(assert_is(thing, "a"))
  expect_silent(assert_is(thing, "b"))
  expect_silent(assert_is(thing, c("a", "b")))
  expect_error(assert_is(thing, "x"),
               "'thing' must be a x")
  expect_error(assert_is(thing, c("x", "y")),
               "'thing' must be a x / y")
})
