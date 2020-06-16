context("utils")

test_that("null-or-value works", {
  expect_equal(1 %||% NULL, 1)
  expect_equal(1 %||% 2, 1)
  expect_equal(NULL %||% NULL, NULL)
  expect_equal(NULL %||% 2, 2)
})


test_that("as_integer/as_size (C++)", {
  expect_error(
    .Call(Ctest_rng, 100, "string", 1L),
    "Expected an integer for 'seed'")
  expect_error(
    .Call(Ctest_rng, 100, 1.5, 1L),
    "Expected an integer for 'seed' (rounding error?)", fixed = TRUE)
  expect_error(
    .Call(Ctest_rng, 100, 1:5, 1L),
    "Expected a scalar for 'seed'", fixed = TRUE)
  expect_error(
    .Call(Ctest_rng, 100, -1, 1L),
    "Expected a non-negative integer for 'seed'", fixed = TRUE)
})


test_that("as_double (C++)", {
  expect_error(
    .Call(Ctest_rng_unif, 100, "string",  1.0,  1L, 1L),
    "Expected a double for 'min'")
  expect_error(
    .Call(Ctest_rng_unif, 100, numeric(3),  1.0,  1L, 1L),
    "Expected a scalar for 'min'", fixed = TRUE)
  expect_identical(
    .Call(Ctest_rng_unif, 100, 0.0, 1L,  1L, 1L),
    .Call(Ctest_rng_unif, 100, 0.0, 1.0,  1L, 1L))
})
