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
  n <- 100
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
  n <- 100
  p <- 0.1

  expect_identical(dust_rng$new(1, 1)$rbinom(m, 0, p),
                   rep(0, m))
  expect_identical(dust_rng$new(1, 1)$rbinom(m, n, 0),
                   rep(0, m))
  expect_identical(dust_rng$new(1, 1)$rbinom(m, n, 1),
                   rep(as.numeric(n), m))
})


test_that("binomial numbers on the 'small' path", {
  m <- 100000
  n <- 20
  p <- 0.2

  ans1 <- dust_rng$new(1, 1)$rbinom(m, n, p)
  expect_equal(mean(ans1), n * p, tolerance = 1e-3)
  expect_equal(var(ans1), n * p * (1 - p), tolerance = 1e-2)
})


test_that("binomial numbers and their complement are the same (np small)", {
  m <- 100
  n <- 20
  p <- 0.2

  ans1 <- dust_rng$new(1, 1)$rbinom(m, n, p)
  ans2 <- dust_rng$new(1, 1)$rbinom(m, n, 1 - p)
  expect_equal(ans1, n - ans2)
})


test_that("binomial numbers and their complement are the same (np large)", {
  m <- 100
  n <- 200
  p <- 0.2

  ans1 <- dust_rng$new(1, 1)$rbinom(m, n, p)
  ans2 <- dust_rng$new(1, 1)$rbinom(m, n, 1 - p)
  expect_equal(ans1, n - ans2)
})


test_that("Binomial random numbers prevent bad inputs", {
  skip_on_cran() # potentially system dependent
  r <- dust_rng$new(1, 1)
  r$rbinom(1, 0, 0)
  expect_error(
    r$rbinom(1, 1, -1),
    "Invalid call to rbinom with n = 1, p = -1")
  expect_error(
    r$rbinom(1, 1, 0 - 1e-8),
    "Invalid call to rbinom with n = 1, p = -1e-08")
  expect_error(
    r$rbinom(1, 1, 2),
    "Invalid call to rbinom with n = 1, p = 2")
  ## TODO: this is not a great error here, but there's not much that
  ## can be done without a lot of faff with the underlying print
  expect_error(
    r$rbinom(1, 1, 1 + 1e-8),
    "Invalid call to rbinom with n = 1, p = 1")
  expect_error(
    r$rbinom(1, -1, 0.5),
    "Invalid call to rbinom with n = -1, p = 0.5")
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


test_that("Short circuit exit does not update rng state", {
  rng <- dust_rng$new(1, 1)
  s <- rng$state()
  ans <- rng$rpois(100, 0)
  expect_equal(ans, rep(0, 100))
  expect_identical(rng$state(), s)
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
  expect_identical(dust_rng$new(rng10$state(), 5L, "float")$state(),
                   rng5$state())
})


test_that("require that raw vector is of sensible size", {
  expect_error(dust_rng$new(raw(), 1L),
               "Expected raw vector of length as multiple of 32 for 'seed'")
  expect_error(dust_rng$new(raw(31), 1L),
               "Expected raw vector of length as multiple of 32 for 'seed'")
  expect_error(dust_rng$new(raw(63), 1L),
               "Expected raw vector of length as multiple of 32 for 'seed'")
  expect_error(dust_rng$new(raw(63), 1L, "float"),
               "Expected raw vector of length as multiple of 32 for 'seed'")
})


test_that("initialise with NULL, generating a seed from R", {
  set.seed(1)
  rng1 <- dust_rng$new(NULL, 1L)
  set.seed(1)
  rng2 <- dust_rng$new(NULL, 1L)
  rng3 <- dust_rng$new(NULL, 1L)
  set.seed(1)
  rng4 <- dust_rng$new(NULL, 1L, "float")
  rng5 <- dust_rng$new(NULL, 1L, "float")

  expect_identical(rng2$state(), rng1$state())
  expect_false(identical(rng3$state(), rng2$state()))

  expect_identical(rng4$state(), rng1$state())
  expect_identical(rng5$state(), rng3$state())
})


test_that("can't create rng with silly things", {
  expect_error(
    dust_rng$new(mtcars, 1L),
    "Invalid type for 'seed'")
  expect_error(
    dust_rng$new(function(x) 2, 1L),
    "Invalid type for 'seed'")
  expect_error(
    dust_rng$new(function(x) 2, 1L, "float"),
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


test_that("binomial random numbers from floats have correct distribution", {
  m <- 100000
  n <- 958
  p <- 0.004145
  yf <- dust_rng$new(1, 1, "float")$rbinom(m, n, p)
  yd <- dust_rng$new(1, 1, "double")$rbinom(m, n, p)
  expect_equal(mean(yf), mean(yd), tolerance = 1e-4)
  expect_equal(var(yf), var(yd), tolerance = 1e-3)
})


test_that("special case", {
  ## This has been fairly carefully selected; with this set of
  ## parameters we get one infinite loop in dust 0.9.7
  m <- 1000000
  n <- 6
  p <- 0.449999988
  yf <- dust_rng$new(1, 1, "float")$rbinom(m, n, p)
  yd <- dust_rng$new(1, 1, "double")$rbinom(m, n, p)

  expect_equal(mean(yf), mean(yd), tolerance = 1e-5)
  expect_equal(var(yf), var(yd), tolerance = 1e-4)
})


test_that("binomial random numbers from floats have correct distribution", {
  m <- 100000
  n <- 100
  p <- 0.1
  yf <- dust_rng$new(1, 1, "float")$rbinom(m, n, p)
  yd <- dust_rng$new(1, 1, "double")$rbinom(m, n, p)
  expect_equal(mean(yf), mean(yd), tolerance = 1e-4)
  expect_equal(var(yf), var(yd), tolerance = 1e-3)
})


test_that("float/double binom identical behaviour in corner cases", {
  rng_f <- dust_rng$new(1, 1, "float")

  ## Short circuiting does not advance rng:
  s <- rng_f$state()
  expect_equal(rng_f$rbinom(100, 0, 0.1), rep(0, 100))
  expect_equal(rng_f$rbinom(100, 5, 0), rep(0, 100))
  expect_equal(rng_f$rbinom(100, 5, 1), rep(5, 100))
  expect_identical(rng_f$state(), s)

  ## ...nor does an error
  expect_error(
    rng_f$rbinom(100, -1, 0.5),
    "Invalid call to rbinom with n = -1, p = 0.5")
  expect_identical(rng_f$state(), s)

  ## and a draw and its complement are the same
  n <- 20
  ans1 <- dust_rng$new(1, 1, "float")$rbinom(100, n, 0.2)
  ans2 <- dust_rng$new(1, 1, "float")$rbinom(100, n, 0.8)
  expect_equal(ans1, n - ans2)
})


test_that("poisson random numbers from floats have correct distribution", {
  n <- 100000
  lambda <- 10
  yf <- dust_rng$new(1, 1, "float")$rpois(n, lambda)
  yd <- dust_rng$new(1, 1, "double")$rpois(n, lambda)
  expect_equal(mean(yf), mean(yd), tolerance = 1e-4)
  expect_equal(var(yf), var(yd), tolerance = 1e-3)
})


test_that("uniform random numbers from floats have correct distribution", {
  n <- 100000
  min <- -2
  max <- 4
  yf <- dust_rng$new(1, 1, "float")$runif(n, min, max)
  yd <- dust_rng$new(1, 1, "double")$runif(n, min, max)
  expect_lt(max(abs(yf - yd)), 1e-6)
})


test_that("normal random numbers from floats have correct distribution", {
  n <- 100000
  mu <- 2
  sd <- 0.1
  yf <- dust_rng$new(1, 1, "float")$rnorm(n, mu, sd)
  yd <- dust_rng$new(1, 1, "double")$rnorm(n, mu, sd)
  expect_lt(max(abs(yf - yd)), 1e-6)
})


test_that("std uniform random numbers from floats have correct distribution", {
  n <- 100000
  yf <- dust_rng$new(1, 1, "float")$unif_rand(n)
  yd <- dust_rng$new(1, 1, "double")$unif_rand(n)
  expect_lt(max(abs(yf - yd)), 1e-6)
})


test_that("std normal random numbers from floats have correct distribution", {
  n <- 100000
  yf <- dust_rng$new(1, 1, "float")$norm_rand(n)
  yd <- dust_rng$new(1, 1, "double")$norm_rand(n)
  expect_lt(max(abs(yf - yd)), 4e-6)
})


test_that("exponential random numbers from floats have correct distribution", {
  n <- 100000
  rate <- 4
  yf <- dust_rng$new(1, 1, "float")$rexp(n, rate)
  yd <- dust_rng$new(1, 1, "double")$rexp(n, rate)
  expect_lt(max(abs(yf - yd)), 1e-6)
})


test_that("float interface works as expected", {
  rng_f <- dust_rng$new(1, 5, "float")
  rng_d <- dust_rng$new(1, 5, "double")
  expect_equal(rng_f$real_type(), "float")
  expect_equal(rng_d$real_type(), "double")
  expect_equal(rng_f$size(), 5L)
  expect_equal(rng_d$size(), 5L)
  expect_identical(rng_f$state(), rng_d$state())
  rng_f$jump()
  rng_d$jump()
  expect_identical(rng_f$state(), rng_d$state())
  rng_f$long_jump()
  rng_d$long_jump()
  expect_identical(rng_f$state(), rng_d$state())
})


test_that("Require double or float", {
  expect_error(
    dust_rng$new(1, 5, "binary16"),
    "Invalid value for 'real_type': must be 'double' or 'float'")
})


test_that("deterministic rbinom returns mean", {
  m <- 10
  n <- as.numeric(sample(10, m, replace = TRUE))
  p <- runif(m)

  rng_f <- dust_rng$new(1, m, "float")
  rng_d <- dust_rng$new(1, m, "double")
  state_f <- rng_f$state()
  state_d <- rng_f$state()

  expect_false(rng_f$set_deterministic(TRUE))
  expect_false(rng_d$set_deterministic(TRUE))

  expect_equal(rng_f$rbinom(m, n, p), n * p, tolerance = 1e-6)
  expect_equal(rng_d$rbinom(m, n, p), n * p)

  expect_true(rng_f$set_deterministic(FALSE))
  expect_true(rng_d$set_deterministic(FALSE))
  expect_equal(rng_f$state(), state_f)
  expect_equal(rng_d$state(), state_d)
})


test_that("deterministic rbinom accepts non-integer size", {
  m <- 10
  n <- runif(m, 0, 10)
  p <- runif(m)
  rng_f <- dust_rng$new(1, m, "float")
  rng_d <- dust_rng$new(1, m, "double")
  state_f <- rng_f$state()
  state_d <- rng_f$state()
  expect_false(rng_f$set_deterministic(TRUE))
  expect_false(rng_d$set_deterministic(TRUE))
  expect_equal(rng_f$rbinom(m, n, p), n * p, tolerance = 1e-6)
  expect_equal(rng_d$rbinom(m, n, p), n * p)
  expect_equal(rng_f$state(), state_f)
  expect_equal(rng_d$state(), state_d)
})


test_that("deterministic rpois returns mean", {
  m <- 10
  lambda <- runif(m, 0, 50)
  rng_f <- dust_rng$new(1, m, "float")
  rng_d <- dust_rng$new(1, m, "double")
  state_f <- rng_f$state()
  state_d <- rng_f$state()
  rng_f$set_deterministic(TRUE)
  rng_d$set_deterministic(TRUE)
  expect_equal(rng_f$rpois(m, lambda), lambda, tolerance = 1e-6)
  expect_equal(rng_d$rpois(m, lambda), lambda)
  expect_equal(rng_f$state(), state_f)
  expect_equal(rng_d$state(), state_d)
})


test_that("deterministic rpois returns mean", {
  m <- 10
  lambda <- runif(m, 0, 50)
  rng_f <- dust_rng$new(1, m, "float")
  rng_d <- dust_rng$new(1, m, "double")
  state_f <- rng_f$state()
  state_d <- rng_f$state()
  rng_f$set_deterministic(TRUE)
  rng_d$set_deterministic(TRUE)
  expect_equal(rng_f$rpois(m, lambda), lambda, tolerance = 1e-6)
  expect_equal(rng_d$rpois(m, lambda), lambda)
  expect_equal(rng_f$state(), state_f)
  expect_equal(rng_d$state(), state_d)
})



test_that("deterministic runif returns mean", {
  m <- 10
  l <- runif(m, -10, 10)
  u <- l + runif(m, 0, 10)
  rng_f <- dust_rng$new(1, m, "float")
  rng_d <- dust_rng$new(1, m, "double")
  state_f <- rng_f$state()
  state_d <- rng_f$state()
  rng_f$set_deterministic(TRUE)
  rng_d$set_deterministic(TRUE)
  expect_equal(rng_f$runif(m, l, u), (l + u) / 2, tolerance = 1e-6)
  expect_equal(rng_d$runif(m, l, u), (l + u) / 2)
  expect_equal(rng_f$state(), state_f)
  expect_equal(rng_d$state(), state_d)
})


test_that("deterministic rexp returns mean", {
  m <- 10
  rate <- runif(m, 0, 10)
  rng_f <- dust_rng$new(1, m, "float")
  rng_d <- dust_rng$new(1, m, "double")
  state_f <- rng_f$state()
  state_d <- rng_f$state()
  rng_f$set_deterministic(TRUE)
  rng_d$set_deterministic(TRUE)
  expect_equal(rng_f$rexp(m, rate), 1 / rate, tolerance = 1e-6)
  expect_equal(rng_d$rexp(m, rate), 1 / rate)
  expect_equal(rng_f$state(), state_f)
  expect_equal(rng_d$state(), state_d)
})


test_that("deterministic rnorm returns mean", {
  m <- 10
  mu <- runif(m, -10, 10)
  sd <- runif(m, 0, 10)
  rng_f <- dust_rng$new(1, m, "float")
  rng_d <- dust_rng$new(1, m, "double")
  state_f <- rng_f$state()
  state_d <- rng_f$state()
  rng_f$set_deterministic(TRUE)
  rng_d$set_deterministic(TRUE)
  expect_equal(rng_f$rnorm(m, mu, sd), mu, tolerance = 1e-6)
  expect_equal(rng_d$rnorm(m, mu, sd), mu)
  expect_equal(rng_f$state(), state_f)
  expect_equal(rng_d$state(), state_d)
})
