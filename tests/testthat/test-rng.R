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

  rng1 <- dust_rng$new(seed, 1L)
  rng2 <- dust_rng$new(seed, 2L)
  rng3 <- dust_rng$new(seed, 4L)
  rng4 <- dust_rng$new(seed, 8L)

  ans1 <- rng1$unif_rand(n)
  ans2 <- rng2$unif_rand(n)
  ans3 <- rng3$unif_rand(n)
  ans4 <- rng4$unif_rand(n)

  ## We can find elements from the first rng through the other
  ## sequences:
  expect_identical(ans1[1:64], ans2[seq(1, 128, by = 2)])
  expect_identical(ans1[1:32], ans3[seq(1, 128, by = 4)])
  expect_identical(ans1[1:16], ans4[seq(1, 128, by = 8)])

  ## The second also appears:
  expect_equal(ans2[seq(2, 64, by = 2)], ans3[seq(2, 128, by = 4)])
  expect_equal(ans2[seq(2, 32, by = 2)], ans4[seq(2, 128, by = 8)])

  expect_equal(rng1$size(), 1)
  expect_equal(rng2$size(), 2)
  expect_equal(rng3$size(), 4)
  expect_equal(rng4$size(), 8)
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


test_that("distribution of uniform numbers", {
  m <- 100000
  a <- exp(1)
  b <- pi
  ans <- dust_rng$new(1, 1)$runif(m, a, b)
  expect_equal(mean(ans), (a + b) / 2, tolerance = 1e-3)
  expect_equal(var(ans), (b - a)^2 / 12, tolerance = 1e-2)
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

  expect_identical(dust_rng$new(1, 1)$rbinom(m, 0L, p),
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


test_that("binomial random numbers from floats have correct distribution", {
  test_rbinom_double <- function(i, m, n, p) {
    dust_rng$new(1, 1)$rbinom(m, n, p)
  }
  m <- 100000
  n <- 958L
  p <- 0.004145
  yf <- test_rbinom_float(1L, m, n, p)
  yd <- test_rbinom_double(1L, m, n, p)
  expect_equal(mean(yf), mean(yd), tolerance = 1e-4)
  expect_equal(var(yf), var(yd), tolerance = 1e-3)
})


test_that("binomial random numbers from floats have correct distribution", {
  test_rbinom_double <- function(i, m, n, p) {
    dust_rng$new(1, 1)$rbinom(m, n, p)
  }
  m <- 100000
  n <- 100L
  p <- 0.1
  yf <- test_rbinom_float(1L, m, n, p)
  yd <- test_rbinom_double(1L, m, n, p)
  expect_equal(mean(yf), mean(yd), tolerance = 1e-4)
  expect_equal(var(yf), var(yd), tolerance = 1e-3)
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


test_that("rnorm agrees with stats::rnorm", {
  n <- 100000
  mu <- exp(1)
  sd <- pi
  ans <- dust_rng$new(2, 1)$rnorm(n, mu, sd)
  expect_equal(mean(ans), mu, tolerance = 1e-2)
  expect_equal(sd(ans), sd, tolerance = 1e-2)
  expect_gt(ks.test(ans, "pnorm", mu, sd)$p.value, 0.1)
})


test_that("rexp agrees with stats::rexp", {
  n <- 100000
  rate <- 0.04
  ans <- dust_rng$new(2, 1)$rexp(n, rate)
  expect_equal(mean(ans), 1 / rate, tolerance = 1e-2)
  expect_equal(var(ans), 1 / rate^2, tolerance = 1e-2)
  expect_gt(ks.test(ans, "pexp", rate)$p.value, 0.1)
})


test_that("continue stream", {
  rng1 <- dust_rng$new(1, 1L)
  rng2 <- dust_rng$new(1, 1L)

  y1 <- rng1$runif(100, 0, 1)
  y2_1 <- rng2$runif(50, 0, 1)
  y2_2 <- rng2$runif(50, 0, 1)
  y2 <- c(y2_1, y2_2)
  expect_identical(y1, y2)
})


test_that("recycle failure", {
  p <- runif(10)
  expect_equal(recycle(p[[1]], 10), rep(p[[1]], 10))
  expect_equal(recycle(p, 10), p)
  expect_error(recycle(p, 5),
               "Invalid length for 'p', expected 1 or 5")
  expect_error(recycle(p, 15),
               "Invalid length for 'p', expected 1 or 15")
})


test_that("jump", {
  seed <- 1
  rng1a <- dust_rng$new(seed, 1L)
  rng1b <- dust_rng$new(seed, 1L)$jump()
  rng2 <- dust_rng$new(seed, 2L)

  r2 <- rng2$unif_rand(20)
  r1a <- rng1a$unif_rand(10)
  r1b <- rng1b$unif_rand(10)

  m1 <- rbind(r1a, r1b, deparse.level = 0)
  m2 <- matrix(r2, 2)
  expect_equal(m1, m2)
})


test_that("long jump", {
  seed <- 1
  rng1 <- dust_rng$new(seed, 1L)
  rng2 <- dust_rng$new(seed, 1L)$jump()
  rng3 <- dust_rng$new(seed, 1L)$long_jump()
  rng4 <- dust_rng$new(seed, 1L)$long_jump()$jump()

  r1 <- rng1$unif_rand(20)
  r2 <- rng2$unif_rand(20)
  r3 <- rng3$unif_rand(20)
  r4 <- rng3$unif_rand(20)

  expect_true(all(r1 != r2))
  expect_true(all(r1 != r3))
  expect_true(all(r1 != r4))
  expect_true(all(r2 != r3))
  expect_true(all(r2 != r4))
  expect_true(all(r3 != r4))
})


test_that("get state", {
  seed <- 1
  rng1 <- dust_rng$new(seed, 1L)
  rng2 <- dust_rng$new(seed, 1L)
  rng3 <- dust_rng$new(seed, 2L)

  s1 <- rng1$state()
  expect_type(s1, "raw")
  expect_equal(length(s1), 32)

  s2 <- rng2$state()
  expect_identical(s2, s1)

  s3 <- rng3$state()
  expect_equal(length(s3), 64)
  expect_identical(s3[seq_len(32)], s1)
  expect_identical(s3[-seq_len(32)], rng2$jump()$state())
})


test_that("initialise single rng with binary state", {
  seed <- 42
  rng1 <- dust_rng$new(seed, 1L)
  state <- rng1$state()
  rng2 <- dust_rng$new(state, 1L)
  expect_identical(rng1$state(), rng2$state())
  r1 <- rng1$unif_rand(10)
  r2 <- rng2$unif_rand(10)
  expect_identical(r1, r2)
  expect_identical(rng1$state(), rng2$state())
})


test_that("initialise parallel rng with binary state", {
  seed <- 42
  rng1 <- dust_rng$new(seed, 5L)
  state <- rng1$state()
  rng2 <- dust_rng$new(state, 5L)
  r1 <- rng1$unif_rand(10)
  r2 <- rng2$unif_rand(10)
  expect_identical(r1, r2)
  expect_identical(rng1$state(), rng2$state())
})


test_that("initialise parallel rng with single binary state and jump", {
  seed <- 42
  rng1 <- dust_rng$new(seed, 1L)
  rng2 <- dust_rng$new(seed, 2L)
  state <- rng1$state()
  rng3 <- dust_rng$new(state, 2L)
  expect_identical(rng3$state(), rng2$state())
})


test_that("initialise parallel rng with binary state and drop", {
  seed <- 42
  rng10 <- dust_rng$new(seed, 10L)
  rng5 <- dust_rng$new(rng10$state(), 5L)
  expect_identical(rng5$state(), rng10$state()[seq_len(5 * 4 * 8)])
})


test_that("require that raw vector is of sensible size", {
  expect_error(dust_rng$new(raw(), 1L),
               "Expected raw vector of length as multiple of 32 for 'seed'")
  expect_error(dust_rng$new(raw(31), 1L),
               "Expected raw vector of length as multiple of 32 for 'seed'")
  expect_error(dust_rng$new(raw(63), 1L),
               "Expected raw vector of length as multiple of 32 for 'seed'")
})


test_that("initialise with NULL, generating a seed from R", {
  set.seed(1)
  rng1 <- dust_rng$new(NULL, 1L)
  set.seed(1)
  rng2 <- dust_rng$new(NULL, 1L)
  rng3 <- dust_rng$new(NULL, 1L)
  expect_identical(rng2$state(), rng1$state())
  expect_false(identical(rng3$state(), rng2$state()))
})


test_that("can't create rng with silly things", {
  expect_error(
    dust_rng$new(mtcars, 1L),
    "Invalid type for 'seed'")
  expect_error(
    dust_rng$new(function(x) 2, 1L),
    "Invalid type for 'seed'")
})


test_that("negative seed values result in sensible state", {
  ## Don't end up with all-zero state, and treat different negative
  ## numbers as different (don't truncate to zero or anything
  ## pathalogical)
  s0 <- dust_rng$new(0, 1L)$state()
  s1 <- dust_rng$new(-1, 1L)$state()
  s10 <- dust_rng$new(-10, 1L)$state()

  expect_false(all(s0 == as.raw(0)))
  expect_false(all(s1 == as.raw(0)))
  expect_false(all(s10 == as.raw(0)))
  expect_false(identical(s0, s1))
  expect_false(identical(s0, s10))
  expect_false(identical(s1, s10))
})


test_that("can jump the rng state with dust_rng_state_long_jump", {
  rng <- dust::dust_rng$new(1)
  state <- rng$state()
  r1 <- rng$long_jump()$state()
  r2 <- rng$long_jump()$state()

  expect_identical(dust_rng_state_long_jump(state), r1)
  expect_identical(dust_rng_state_long_jump(state), r1)
  expect_identical(dust_rng_state_long_jump(state, 2), r2)
})
