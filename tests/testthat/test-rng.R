context("rng")

test_that("can generate random numbers", {
  ans1 <- .Call(Ctest_rng, 100L, 1L, 1L)
  ans2 <- .Call(Ctest_rng, 100L, 1L, 1L)
  ans3 <- .Call(Ctest_rng, 100L, 2L, 1L)
  expect_equal(length(ans1), 100)
  expect_identical(ans1, ans2)
  expect_false(any(ans1 == ans3))
})


test_that("Create interleaved rng", {
  n <- 128
  seed <- 1

  ans1 <- .Call(Ctest_rng, n, seed, 1L)
  ans2 <- .Call(Ctest_rng, n, seed, 2L)
  ans3 <- .Call(Ctest_rng, n, seed, 4L)
  ans4 <- .Call(Ctest_rng, n, seed, 8L)

  ## We can find elements from the first rng through the other
  ## sequences:
  expect_identical(ans1[1:64], ans2[seq(1, 128, by = 2)])
  expect_identical(ans1[1:32], ans3[seq(1, 128, by = 4)])
  expect_identical(ans1[1:16], ans4[seq(1, 128, by = 8)])

  ## The second also appears:
  expect_equal(ans2[seq(2, 64, by = 2)], ans3[seq(2, 128, by = 4)])
  expect_equal(ans2[seq(2, 32, by = 2)], ans4[seq(2, 128, by = 8)])
})


test_that("run uniform random numbers", {
  ans1 <- .Call(Ctest_rng_unif, 100, NULL, NULL, 1L, 1L)
  ans2 <- .Call(Ctest_rng_unif, 100, NULL, NULL, 1L, 1L)
  ans3 <- .Call(Ctest_rng_unif, 100, 0.0,  1.0,  1L, 1L)
  ans4 <- .Call(Ctest_rng_unif, 100, NULL, NULL, 2L, 1L)
  expect_true(all(ans1 >= 0))
  expect_true(all(ans1 <= 1))
  expect_identical(ans1, ans2)
  expect_identical(ans1, ans3)
  expect_false(any(ans1 == ans4))
})


test_that("run binomial random numbers", {
  m <- 1000000
  n <- 100L
  p <- 0.1

  nn <- rep(n, m)
  pp <- rep(p, m)

  ans1 <- .Call(Ctest_rng_binom, nn, pp, 1L, 1L)
  ans2 <- .Call(Ctest_rng_binom, nn, pp, 1L, 1L)
  expect_identical(ans1, ans2)

  ## Should do this with much more statistical rigour, but this looks
  ## pretty good.
  expect_equal(mean(ans1), n * p, tolerance = 1e-3)
  expect_equal(var(ans1), n * p * (1 - p), tolerance = 1e-2)
})


test_that("binomial numbers run the short circuit path", {
  m <- 1000000
  n <- 100L
  p <- 0.1
  nn <- rep(n, m)
  pp <- rep(p, m)
  expect_identical(.Call(Ctest_rng_binom, rep(0L, m), pp, 1L, 1L),
                   rep(0L, m))
  expect_identical(.Call(Ctest_rng_binom, nn, rep(0, m), 1L, 1L),
                   rep(0L, m))
  expect_identical(.Call(Ctest_rng_binom, nn, rep(1, m), 1L, 1L),
                   rep(n, m))
})
