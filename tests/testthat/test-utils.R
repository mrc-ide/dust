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


test_that("Can avoid debug in compile_dll", {
  skip_if_not_installed("mockery")

  mock_has_user_makevars <- mockery::mock(FALSE)
  mock_compile_dll <- mockery::mock(
    list(Sys.getenv("R_MAKEVARS_USER"), pkgbuild:::makevars_user()))

  path <- tempfile()
  compile_attributes <- TRUE
  quiet <- FALSE
  res <- with_mock(
    "dust::has_user_makevars" = mock_has_user_makevars,
    "pkgbuild::compile_dll" = mock_compile_dll,
    compile_dll(path, compile_attributes, quiet))

  expect_equal(res[[1]], res[[2]])
  expect_equal(normalizePath(dirname(res[[1]])),
               normalizePath(tempdir()))

  mockery::expect_called(mock_has_user_makevars, 1)
  mockery::expect_called(mock_compile_dll, 1)
  expect_equal(
    mockery::mock_args(mock_compile_dll)[[1]],
    list(path, compile_attributes, quiet))
})


test_that("Don't set envvar if not needed", {
  skip_if_not_installed("mockery")

  env <- c("R_MAKEVARS_USER" = NA)
  cmp <- withr::with_envvar(
    env,
    pkgbuild:::makevars_user())

  mock_has_user_makevars <- mockery::mock(TRUE)
  mock_compile_dll <- mockery::mock(
    list(Sys.getenv("R_MAKEVARS_USER"), pkgbuild:::makevars_user()))

  path <- tempfile()
  compile_attributes <- TRUE
  quiet <- FALSE

  res <- withr::with_envvar(
    env,
    with_mock(
      "dust::has_user_makevars" = mock_has_user_makevars,
      "pkgbuild::compile_dll" = mock_compile_dll,
      compile_dll(path, compile_attributes, quiet)))

  expect_equal(res[[1]], "")
  expect_equal(res[[2]], cmp)

  mockery::expect_called(mock_has_user_makevars, 1)
  mockery::expect_called(mock_compile_dll, 1)
  expect_equal(
    mockery::mock_args(mock_compile_dll)[[1]],
    list(path, compile_attributes, quiet))
})
