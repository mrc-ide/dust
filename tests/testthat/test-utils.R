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
  p <- .Call(Ctest_walk_alloc, 1, 10L, 1L)
  null <- unserialize(serialize(p, NULL))
  expect_error(
    .Call(Ctest_walk_run, null, 1L),
    "Pointer has been invalidated (perhaps serialised?)",
    fixed = TRUE)
  expect_error(
    .Call(Ctest_walk_run, NULL, 1L),
    "Expected an external pointer",
    fixed = TRUE)
})
