context("rng")

test_that("can generate random numbers", {
  ans1 <- dust_rng$new(1, 1)$unif_rand(100)
  ans2 <- dust_rng$new(1, 1)$unif_rand(100)
  ans3 <- dust_rng$new(2, 1)$unif_rand(100)
  expect_equal(length(ans1), 100)
  expect_identical(ans1, ans2)
  expect_false(any(ans1 == ans3))
})


test_that("Create interleaved rng", {
  n <- 128
  seed <- 1

  ans1 <- dust_rng$new(seed, 1L)$unif_rand(n)
  ans2 <- dust_rng$new(seed, 2L)$unif_rand(n)
  ans3 <- dust_rng$new(seed, 4L)$unif_rand(n)
  ans4 <- dust_rng$new(seed, 8L)$unif_rand(n)

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
  ans1 <- dust_rng$new(1L, 1L)$unif_rand(100)
  ans2 <- dust_rng$new(1L, 1L)$unif_rand(100)
  ans3 <- dust_rng$new(1L, 1L)$runif(100, 0, 1)
  ans4 <- dust_rng$new(2L, 1L)$runif(100, 0, 1)

  expect_true(all(ans1 >= 0))
  expect_true(all(ans1 <= 1))
  expect_identical(ans1, ans2)
  expect_identical(ans1, ans3)
  expect_false(any(ans1 == ans4))
})


test_that("run uniform random numbers with odd bounds", {
  ans <- dust_rng$new(1L, 1L)$runif(100, -100, 100)
  expect_true(any(ans > 0))
  expect_true(any(ans < 0))
  expect_true(all(ans >= -100))
  expect_true(all(ans <= 100))
})


test_that("run binomial random numbers", {
  m <- 100000
  n <- 100L
  p <- 0.1

  ans1 <- dust_rng$new(1, 1)$rbinom(m, n, p)
  ans2 <- dust_rng$new(1, 1)$rbinom(m, n, p)
  expect_identical(ans1, ans2)

  ## Should do this with much more statistical rigour, but this looks
  ## pretty good.
  expect_equal(mean(ans1), n * p, tolerance = 1e-3)
  expect_equal(var(ans1), n * p * (1 - p), tolerance = 1e-2)
})


test_that("binomial numbers run the short circuit path", {
  m <- 10000
  n <- 100L
  p <- 0.1

  expect_identical(dust_rng$new(1, 1)$rbinom(m, 0, p),
                   rep(0L, m))
  expect_identical(dust_rng$new(1, 1)$rbinom(m, n, 0),
                   rep(0L, m))
  expect_identical(dust_rng$new(1, 1)$rbinom(m, n, 1),
                   rep(n, m))
})


test_that("binomial numbers on the 'small' path", {
  m <- 100000
  n <- 20L
  p <- 0.2

  ans1 <- dust_rng$new(1, 1)$rbinom(m, n, p)
  expect_equal(mean(ans1), n * p, tolerance = 1e-3)
  expect_equal(var(ans1), n * p * (1 - p), tolerance = 1e-2)
})


test_that("binomial numbers and their complement are the same (np small)", {
  m <- 100
  n <- 20L
  p <- 0.2

  ans1 <- dust_rng$new(1, 1)$rbinom(m, n, p)
  ans2 <- dust_rng$new(1, 1)$rbinom(m, n, 1 - p)
  expect_equal(ans1, n - ans2)
})


test_that("binomial numbers and their complement are the same (np large)", {
  m <- 100
  n <- 200L
  p <- 0.2

  ans1 <- dust_rng$new(1, 1)$rbinom(m, n, p)
  ans2 <- dust_rng$new(1, 1)$rbinom(m, n, 1 - p)
  expect_equal(ans1, n - ans2)
})


test_that("poisson numbers", {
  n <- 100000
  lambda <- 5

  ans1 <- dust_rng$new(1, 1)$rpois(n, lambda)
  ans2 <- dust_rng$new(1, 1)$rpois(n, lambda)
  ans3 <- dust_rng$new(2, 1)$rpois(n, lambda)
  expect_identical(ans1, ans2)
  expect_false(all(ans1 == ans3))

  expect_equal(mean(ans1), lambda, 1e-2)
  expect_equal(var(ans1), lambda, 1e-2)
})


test_that("Big poisson numbers", {
  n <- 100000
  lambda <- 20

  ans1 <- dust_rng$new(1, 1)$rpois(n, lambda)
  ans2 <- dust_rng$new(1, 1)$rpois(n, lambda)
  ans3 <- dust_rng$new(2, 1)$rpois(n, lambda)
  expect_identical(ans1, ans2)
  expect_false(all(ans1 == ans3))

  expect_equal(mean(ans1), lambda, 1e-2)
  expect_equal(var(ans1), lambda, 1e-2)
})


test_that("norm_rand agrees with rnorm", {
  n <- 100000
  ans <- dust_rng$new(2, 1)$norm_rand(n)
  expect_equal(mean(ans), 0, tolerance = 1e-2)
  expect_equal(var(ans), 1, tolerance = 1e-2)
  expect_gt(ks.test(ans, "pnorm")$p.value, 0.1)
})
