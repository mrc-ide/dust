context("example")

test_that("create walk", {
  w <- .Call(Ctest_walk, 1, 10, 1)
})
