context("example")

test_that("create walk, stepping for one step", {
  p <- .Call(Ctest_walk_alloc, 1, 10L, 1L)
  expect_is(p, "externalptr")

  res <- .Call(Ctest_walk_run, p, 1L)
  cmp <- .Call(Ctest_rng, 10L, 1L)
  expect_identical(drop(res), cmp)
})


test_that("walk agrees with random number stream", {
  p <- .Call(Ctest_walk_alloc, 1, 10L, 1L)
  expect_is(p, "externalptr")

  res <- .Call(Ctest_walk_run, p, 5L)
  cmp <- .Call(Ctest_rng, 50L, 1L)
  expect_equal(drop(res), colSums(matrix(cmp, 5, 10)))
})
