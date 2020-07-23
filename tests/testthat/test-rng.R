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
