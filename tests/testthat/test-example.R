context("example")

test_that("create walk", {
  p <- .Call(Ctest_walk_alloc, 1, 10, 1)
  expect_is(p, "externalptr")

  .Call(Ctest_walk_run, p, 10)
})
