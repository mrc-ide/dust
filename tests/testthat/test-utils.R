context("utils")

test_that("null-or-value works", {
  expect_equal(1 %||% NULL, 1)
  expect_equal(1 %||% 2, 1)
  expect_equal(NULL %||% NULL, NULL)
  expect_equal(NULL %||% 2, 2)
})


test_that("as_integer/as_size (C++)", {
  expect_error(
    .Call(Ctest_rng, 100, "string"),
    "Expected an integer for 'seed'")
  expect_error(
    .Call(Ctest_rng, 100, 1.5),
    "Expected an integer for 'seed' (rounding error?)", fixed = TRUE)
  expect_error(
    .Call(Ctest_rng, 100, 1:5),
    "Expected a scalar for 'seed'", fixed = TRUE)
  expect_error(
    .Call(Ctest_rng, 100, -1),
    "Expected a non-negative integer for 'seed'", fixed = TRUE)
})


test_that("check pointer", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "my_walk")

  obj <- res$new(1, 0, 10, 1)
  private <- r6_private(obj)

  private$ptr <- unserialize(serialize(private$ptr, NULL))
  expect_error(
    obj$run(10),
    "Pointer has been invalidated (perhaps serialised?)",
    fixed = TRUE)

  private$ptr <- NULL
  expect_error(
    obj$run(10),
    "Expected an external pointer",
    fixed = TRUE)
})


test_that("compilation caching works", {
  path <- tempfile()
  dir.create(path)
  dest <- file.path(path, "hello.c")
  writeLines(c(
    "int add(int a, int b) {",
    "  return a + b;",
    "}"),
    dest)
  expect_message(
    res <- compile(dest),
    "Compiling shared library")
  expect_true(file.exists(res$dll))

  file.create(res$dll) # truncates file
  expect_message(
    res <- compile(dest),
    "Using previously compiled shared library")
  expect_equal(file.size(res$dll), 0) # unchanged
})


test_that("compilation failing throws error", {
  path <- tempfile()
  dir.create(path)
  dest <- file.path(path, "hello.c")
  writeLines(c(
    "int add(int a, int b) {",
    "  return a + b;"),
    dest)
  expect_error(
    compile(dest),
    "Error compiling source")
})
