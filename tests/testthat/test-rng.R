context("rng")

test_that("can generate random numbers", {
  ans1 <- .Call(Ctest_rng, 100L, 1L)
  ans2 <- .Call(Ctest_rng, 100L, 1L)
  ans3 <- .Call(Ctest_rng, 100L, 2L)
  expect_equal(length(ans1), 100)
  expect_identical(ans1, ans2)
  expect_false(any(ans1 == ans3))
})
