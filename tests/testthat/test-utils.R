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


test_that("writelines_if_changed doesn't replace file", {
  text1 <- c("a", "b", "c")
  text2 <- c("a", "b", "c", "d")
  str1 <- paste(text1, collapse = "\n")
  str2 <- structure(str1, class = "glue")
  path <- tempfile()
  writelines_if_changed(text1, path)
  expect_true(same_content(path, text1))
  expect_true(same_content(path, str1))
  expect_true(same_content(path, str2))
  writelines_if_changed(str1, path)
  writelines_if_changed(str2, path)
  expect_true(file.exists(path))
  expect_equal(readLines(path), text1)
  t <- file.mtime(path)
  writelines_if_changed(text1, path)
  expect_identical(file.mtime(path), t)
  writelines_if_changed(text2, path)
  expect_equal(readLines(path), text2)
  ## I don't trust times and sub-second accuracy not guaranted; see
  ## ?file.mtime
  skip_on_cran()
  skip_on_os("windows")
  expect_gt(file.mtime(path), t)
})


test_that("simple cache allows skipping", {
  obj <- simple_cache$new()
  key <- "a"
  value <- runif(10)
  expect_false(obj$has_key(key, FALSE))
  expect_false(obj$has_key(key, TRUE))
  expect_null(obj$get(key, FALSE))
  expect_null(obj$get(key, TRUE))

  ## set with skip does nothing
  obj$set(key, value, TRUE)
  expect_false(obj$has_key(key, FALSE))
  expect_false(obj$has_key(key, TRUE))
  expect_null(obj$get(key, FALSE))
  expect_null(obj$get(key, TRUE))

  ## set without skip adds key
  obj$set(key, value, FALSE)
  expect_true(obj$has_key(key, FALSE))
  expect_false(obj$has_key(key, TRUE))
  expect_equal(obj$get(key, FALSE), value)
  expect_null(obj$get(key, TRUE))
})
