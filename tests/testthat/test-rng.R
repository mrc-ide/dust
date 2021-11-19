test_that("can generate random numbers", {
  ans1 <- dust_rng$new(1)$random_real(100)
  ans2 <- dust_rng$new(1)$random_real(100)
  ans3 <- dust_rng$new(2)$random_real(100)
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

  ans1 <- rng1$random_real(n)
  ans2 <- rng2$random_real(n)
  ans3 <- rng3$random_real(n)
  ans4 <- rng4$random_real(n)

  ## We can find elements from the each rng through the larger
  ## sequences:
  expect_identical(ans1, ans2[, 1])
  expect_identical(ans1, ans3[, 1])
  expect_identical(ans1, ans3[, 1])
  expect_identical(ans2, ans3[, 1:2])
  expect_identical(ans2, ans4[, 1:2])
  expect_identical(ans3, ans4[, 1:4])

  expect_equal(rng1$size(), 1)
  expect_equal(rng2$size(), 2)
  expect_equal(rng3$size(), 4)
  expect_equal(rng4$size(), 8)
})


test_that("run uniform random numbers", {
  ans1 <- dust_rng$new(1L)$random_real(100)
  ans2 <- dust_rng$new(1L)$random_real(100)
  ans3 <- dust_rng$new(1L)$uniform(100, 0, 1)
  ans4 <- dust_rng$new(2L)$uniform(100, 0, 1)

  expect_true(all(ans1 >= 0))
  expect_true(all(ans1 <= 1))
  expect_identical(ans1, ans2)
  expect_identical(ans1, ans3)
  expect_false(any(ans1 == ans4))
})


test_that("run uniform random numbers with odd bounds", {
  ans <- dust_rng$new(1L)$uniform(100, -100, 100)
  expect_true(any(ans > 0))
  expect_true(any(ans < 0))
  expect_true(all(ans >= -100))
  expect_true(all(ans <= 100))
})


test_that("distribution of uniform numbers", {
  m <- 100000
  a <- exp(1)
  b <- pi
  ans <- dust_rng$new(1)$uniform(m, a, b)
  expect_equal(mean(ans), (a + b) / 2, tolerance = 1e-3)
  expect_equal(var(ans), (b - a)^2 / 12, tolerance = 1e-2)
})


test_that("run binomial random numbers", {
  m <- 100000
  n <- 100
  p <- 0.1

  ans1 <- dust_rng$new(1)$binomial(m, n, p)
  ans2 <- dust_rng$new(1)$binomial(m, n, p)
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

  expect_identical(dust_rng$new(1)$binomial(m, 0, p),
                   rep(0, m))
  expect_identical(dust_rng$new(1)$binomial(m, n, 0),
                   rep(0, m))
  expect_identical(dust_rng$new(1)$binomial(m, n, 1),
                   rep(as.numeric(n), m))
})


test_that("binomial numbers on the 'small' path", {
  m <- 500000
  n <- 20
  p <- 0.2

  ans1 <- dust_rng$new(1)$binomial(m, n, p)
  expect_equal(mean(ans1), n * p, tolerance = 1e-3)
  expect_equal(var(ans1), n * p * (1 - p), tolerance = 1e-2)
})


test_that("binomial numbers and their complement are the same (np small)", {
  m <- 100
  n <- 20
  p <- 0.2

  ans1 <- dust_rng$new(1)$binomial(m, n, p)
  ans2 <- dust_rng$new(1)$binomial(m, n, 1 - p)
  expect_equal(ans1, n - ans2)
})


test_that("binomial numbers and their complement are the same (np large)", {
  m <- 100
  n <- 200
  p <- 0.2

  ans1 <- dust_rng$new(1)$binomial(m, n, p)
  ans2 <- dust_rng$new(1)$binomial(m, n, 1 - p)
  expect_equal(ans1, n - ans2)
})


test_that("Binomial random numbers prevent bad inputs", {
  skip_on_cran() # potentially system dependent
  r <- dust_rng$new(1)
  r$binomial(1, 0, 0)
  expect_error(
    r$binomial(1, 1, -1),
    "Invalid call to binomial with n = 1, p = -1")
  expect_error(
    r$binomial(1, 1, 0 - 1e-8),
    "Invalid call to binomial with n = 1, p = -1e-08")
  expect_error(
    r$binomial(1, 1, 2),
    "Invalid call to binomial with n = 1, p = 2")
  ## TODO: this is not a great error here, but there's not much that
  ## can be done without a lot of faff with the underlying print
  expect_error(
    r$binomial(1, 1, 1 + 1e-8),
    "Invalid call to binomial with n = 1, p = 1")
  expect_error(
    r$binomial(1, -1, 0.5),
    "Invalid call to binomial with n = -1, p = 0.5")
})


test_that("poisson numbers", {
  n <- 100000
  lambda <- 5

  ans1 <- dust_rng$new(1)$poisson(n, lambda)
  ans2 <- dust_rng$new(1)$poisson(n, lambda)
  ans3 <- dust_rng$new(2)$poisson(n, lambda)
  expect_identical(ans1, ans2)
  expect_false(all(ans1 == ans3))

  expect_equal(mean(ans1), lambda, tolerance = 1e-2)
  expect_equal(var(ans1), lambda, tolerance = 1e-2)
})


test_that("Big poisson numbers", {
  n <- 100000
  lambda <- 20

  ans1 <- dust_rng$new(1)$poisson(n, lambda)
  ans2 <- dust_rng$new(1)$poisson(n, lambda)
  ans3 <- dust_rng$new(2)$poisson(n, lambda)
  expect_identical(ans1, ans2)
  expect_false(all(ans1 == ans3))

  expect_equal(mean(ans1), lambda, tolerance = 1e-2)
  expect_equal(var(ans1), lambda, tolerance = 1e-2)
})


test_that("Short circuit exit does not update rng state", {
  rng <- dust_rng$new(1)
  s <- rng$state()
  ans <- rng$poisson(100, 0)
  expect_equal(ans, rep(0, 100))
  expect_identical(rng$state(), s)
})


test_that("normal (box_muller) agrees with stats::rnorm", {
  n <- 100000
  ans <- dust_rng$new(2)$random_normal(n)
  expect_equal(mean(ans), 0, tolerance = 1e-2)
  expect_equal(sd(ans), 1, tolerance = 1e-2)
  expect_gt(ks.test(ans, "pnorm")$p.value, 0.1)
})


test_that("normal (polar) agrees with stats::rnorm", {
  n <- 100000
  ans <- dust_rng$new(2)$random_normal(n, algorithm = "polar")
  expect_equal(mean(ans), 0, tolerance = 1e-2)
  expect_equal(sd(ans), 1, tolerance = 1e-2)
  expect_gt(ks.test(ans, "pnorm")$p.value, 0.1)
})


test_that("normal (ziggurat) agrees with stats::rnorm", {
  n <- 100000
  ans <- dust_rng$new(2)$random_normal(n, algorithm = "ziggurat")
  expect_equal(mean(ans), 0, tolerance = 1e-2)
  expect_equal(sd(ans), 1, tolerance = 1e-2)
  expect_gt(ks.test(ans, "pnorm")$p.value, 0.1)
})


test_that("normal scales draws", {
  n <- 100
  mean <- exp(1)
  sd <- pi
  rng1 <- dust_rng$new(1)
  rng2 <- dust_rng$new(1)
  expect_equal(rng1$normal(n, mean, sd),
               mean + sd * rng2$random_normal(n))
  expect_equal(rng1$normal(n, mean, sd, algorithm = "polar"),
               mean + sd * rng2$random_normal(n, algorithm = "polar"))
  expect_equal(rng1$normal(n, mean, sd, algorithm = "ziggurat"),
               mean + sd * rng2$random_normal(n, algorithm = "ziggurat"))
})


test_that("Prevent unknown normal algorithms", {
  expect_error(
    dust_rng$new(2)$random_normal(10, algorithm = "monty_python"),
    "Unknown normal algorithm 'monty_python'")
  expect_error(
    dust_rng$new(2)$normal(10, 0, 1, algorithm = "monty_python"),
    "Unknown normal algorithm 'monty_python'")
})


test_that("rexp agrees with stats::rexp", {
  n <- 100000
  rate <- 0.04
  ans <- dust_rng$new(2)$exponential(n, rate)
  expect_equal(mean(ans), 1 / rate, tolerance = 1e-2)
  expect_equal(var(ans), 1 / rate^2, tolerance = 1e-2)
  expect_gt(ks.test(ans, "pexp", rate)$p.value, 0.1)
})


test_that("continue stream", {
  rng1 <- dust_rng$new(1)
  rng2 <- dust_rng$new(1)

  y1 <- rng1$uniform(100, 0, 1)
  y2_1 <- rng2$uniform(50, 0, 1)
  y2_2 <- rng2$uniform(50, 0, 1)
  y2 <- c(y2_1, y2_2)
  expect_identical(y1, y2)
})


test_that("jump", {
  seed <- 1
  rng1a <- dust_rng$new(seed)
  rng1b <- dust_rng$new(seed)$jump()
  rng2 <- dust_rng$new(seed, 2L)

  r2 <- rng2$random_real(10)
  r1a <- rng1a$random_real(10)
  r1b <- rng1b$random_real(10)

  expect_equal(cbind(r1a, r1b, deparse.level = 0), r2)
})


test_that("long jump", {
  seed <- 1
  rng1 <- dust_rng$new(seed)
  rng2 <- dust_rng$new(seed)$jump()
  rng3 <- dust_rng$new(seed)$long_jump()
  rng4 <- dust_rng$new(seed)$long_jump()$jump()

  r1 <- rng1$random_real(20)
  r2 <- rng2$random_real(20)
  r3 <- rng3$random_real(20)
  r4 <- rng4$random_real(20)

  expect_true(all(r1 != r2))
  expect_true(all(r1 != r3))
  expect_true(all(r1 != r4))
  expect_true(all(r2 != r3))
  expect_true(all(r2 != r4))
  expect_true(all(r3 != r4))
})


test_that("get state", {
  seed <- 1
  rng1 <- dust_rng$new(seed)
  rng2 <- dust_rng$new(seed)
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
  rng1 <- dust_rng$new(seed)
  state <- rng1$state()
  rng2 <- dust_rng$new(state)
  expect_identical(rng1$state(), rng2$state())
  r1 <- rng1$random_real(10)
  r2 <- rng2$random_real(10)
  expect_identical(r1, r2)
  expect_identical(rng1$state(), rng2$state())
})


test_that("initialise parallel rng with binary state", {
  seed <- 42
  rng1 <- dust_rng$new(seed, 5L)
  state <- rng1$state()
  rng2 <- dust_rng$new(state, 5L)
  r1 <- rng1$random_real(10)
  r2 <- rng2$random_real(10)
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
  len <- 5 * rng5$info$size_state_bytes
  expect_identical(rng5$state(), rng10$state()[seq_len(len)])
})


test_that("initialise parallel rng with binary state and drop for floats", {
  seed <- 42
  rng10 <- dust_rng$new(seed, 10L, "float")
  rng5 <- dust_rng$new(rng10$state(), 5L, "float")
  len <- 5 * rng5$info$size_state_bytes
  expect_identical(rng5$state(), rng10$state()[seq_len(len)])
})


test_that("require that raw vector is of sensible size", {
  expect_error(dust_rng$new(raw()),
               "Expected raw vector of length as multiple of 32 for 'seed'")
  expect_error(dust_rng$new(raw(31)),
               "Expected raw vector of length as multiple of 32 for 'seed'")
  expect_error(dust_rng$new(raw(63)),
               "Expected raw vector of length as multiple of 32 for 'seed'")
  expect_error(dust_rng$new(raw(63), real_type = "float"),
               "Expected raw vector of length as multiple of 16 for 'seed'")
})


test_that("initialise with NULL, generating a seed from R", {
  set.seed(1)
  rng1 <- dust_rng$new(NULL)
  set.seed(1)
  rng2 <- dust_rng$new(NULL)
  rng3 <- dust_rng$new(NULL)
  set.seed(1)
  rng4 <- dust_rng$new(NULL, real_type = "float")
  rng5 <- dust_rng$new(NULL, real_type = "float")

  expect_identical(rng2$state(), rng1$state())
  expect_false(identical(rng3$state(), rng2$state()))

  i <- rep(rep(c(TRUE, FALSE), each = 4), 4)
  expect_identical(rng4$state(), rng1$state()[i])
  expect_identical(rng5$state(), rng3$state()[i])
})


test_that("can't create rng with silly things", {
  expect_error(
    dust_rng$new(mtcars),
    "Invalid type for 'seed'")
  expect_error(
    dust_rng$new(function(x) 2),
    "Invalid type for 'seed'")
  expect_error(
    dust_rng$new(function(x) 2, real_type = "float"),
    "Invalid type for 'seed'")
})


test_that("negative seed values result in sensible state", {
  ## Don't end up with all-zero state, and treat different negative
  ## numbers as different (don't truncate to zero or anything
  ## pathalogical)
  s0 <- dust_rng$new(0)$state()
  s1 <- dust_rng$new(-1)$state()
  s10 <- dust_rng$new(-10)$state()

  expect_false(all(s0 == as.raw(0)))
  expect_false(all(s1 == as.raw(0)))
  expect_false(all(s10 == as.raw(0)))
  expect_false(identical(s0, s1))
  expect_false(identical(s0, s10))
  expect_false(identical(s1, s10))
})


test_that("binomial random numbers from floats have correct distribution", {
  m <- 1000000
  n <- 958
  p <- 0.004145
  yf <- dust_rng$new(1, real_type = "float")$binomial(m, n, p)
  expect_equal(mean(yf), n * p, tolerance = 1e-3)
  expect_equal(var(yf), n * p * (1 - p), tolerance = 1e-2)
})


test_that("special case", {
  ## This has been fairly carefully selected; with this set of
  ## parameters we get one infinite loop in dust 0.9.7
  m <- 1000000
  n <- 6
  p <- 0.449999988
  yf <- dust_rng$new(1, real_type = "float")$binomial(m, n, p)

  expect_equal(mean(yf), n * p, tolerance = 1e-3)
  expect_equal(var(yf), n * p * (1 - p), tolerance = 1e-2)
})


test_that("binomial random numbers from floats have correct distribution", {
  m <- 100000
  n <- 100
  p <- 0.1
  yf <- dust_rng$new(1, real_type = "float")$binomial(m, n, p)
  expect_equal(mean(yf), n * p, tolerance = 1e-2)
  expect_equal(var(yf), n * p * (1 - p), tolerance = 1e-2)
})


test_that("float/double binom identical behaviour in corner cases", {
  rng_f <- dust_rng$new(1, real_type = "float")

  ## Short circuiting does not advance rng:
  s <- rng_f$state()
  expect_equal(rng_f$binomial(100, 0, 0.1), rep(0, 100))
  expect_equal(rng_f$binomial(100, 5, 0), rep(0, 100))
  expect_equal(rng_f$binomial(100, 5, 1), rep(5, 100))
  expect_identical(rng_f$state(), s)

  ## ...nor does an error
  expect_error(
    rng_f$binomial(100, -1, 0.5),
    "Invalid call to binomial with n = -1, p = 0.5")
  expect_identical(rng_f$state(), s)

  ## and a draw and its complement are the same
  n <- 20
  ans1 <- dust_rng$new(1, real_type = "float")$binomial(100, n, 0.2)
  ans2 <- dust_rng$new(1, real_type = "float")$binomial(100, n, 0.8)
  expect_equal(ans1, n - ans2)
})


test_that("poisson random numbers from floats have correct distribution", {
  n <- 100000
  lambda <- 10
  yf <- dust_rng$new(1, real_type = "float")$poisson(n, lambda)
  expect_equal(mean(yf), lambda, tolerance = 1e-3)
  expect_equal(var(yf), lambda, tolerance = 5e-3)
})


test_that("uniform random numbers from floats have correct distribution", {
  n <- 100000
  min <- -2
  max <- 4
  yf <- dust_rng$new(1, real_type = "float")$uniform(n, min, max)
  expect_equal(mean(yf), (min + max) / 2, tolerance = 1e-2)
  expect_equal(var(yf), (max - min)^2 / 12, tolerance = 1e-2)
})


test_that("normal random numbers from floats have correct distribution", {
  n <- 100000
  y <- dust_rng$new(1, real_type = "float")$random_normal(n)
  expect_equal(mean(y), 0, tolerance = 1e-2)
  expect_equal(sd(y), 1, tolerance = 1e-2)
  expect_gt(suppressWarnings(ks.test(y, "pnorm")$p.value), 0.1)

  m <- 200
  mu <- exp(1)
  sd <- pi
  expect_equal(
    dust_rng$new(1, real_type = "float")$normal(m, mu, sd),
    mu + sd * y[seq_len(m)],
    tolerance = 1e-5)
})


test_that("normal draws from floats have correct distribution (polar)", {
  n <- 100000
  r <- dust_rng$new(1, real_type = "float")
  y <- r$random_normal(n, algorithm = "polar")
  expect_equal(mean(y), 0, tolerance = 1e-2)
  expect_equal(sd(y), 1, tolerance = 1e-2)
  expect_gt(suppressWarnings(ks.test(y, "pnorm")$p.value), 0.1)

  cmp <- dust_rng$new(1, real_type = "float")
  m <- 200
  mu <- exp(1)
  sd <- pi
  expect_equal(
    cmp$normal(m, mu, sd, algorithm = "polar"),
    mu + sd * y[seq_len(m)],
    tolerance = 1e-5)
})


test_that("normal random numbers from floats have correct distribution (zig)", {
  ## Reordering the two draws used in the ziggrat algorithm here
  ## created a mild failure that is not sytematic (p value of 0.04);
  ## however the subsequent set of draws were not failures so this is
  ## likely fine.
  n <- 100000
  r <- dust_rng$new(2, real_type = "float")
  y <- r$random_normal(n, algorithm = "ziggurat")
  expect_equal(mean(y), 0, tolerance = 1e-2)
  expect_equal(sd(y), 1, tolerance = 1e-2)
  expect_gt(suppressWarnings(ks.test(y, "pnorm")$p.value), 0.1)

  cmp <- dust_rng$new(2, real_type = "float")
  m <- 200
  mu <- exp(1)
  sd <- pi
  expect_equal(
    cmp$normal(m, mu, sd, algorithm = "ziggurat"),
    mu + sd * y[seq_len(m)],
    tolerance = 1e-5)
})


test_that("std uniform random numbers from floats have correct distribution", {
  n <- 100000
  yf <- dust_rng$new(42, real_type = "float")$random_real(n)
  expect_equal(mean(yf), 0.5, tolerance = 1e-3)
  expect_equal(var(yf), 1 / 12, tolerance = 1e-2)
})


test_that("exponential random numbers from floats have correct distribution", {
  n <- 100000
  rate <- 4
  yf <- dust_rng$new(1, real_type = "float")$exponential(n, rate)
  expect_equal(mean(yf), 1 / rate, tolerance = 1e-2)
  expect_equal(var(yf), 1 / rate^2, tolerance = 5e-2)
  expect_gt(suppressWarnings(ks.test(yf, "pexp", rate)$p.value), 0.1)
})


test_that("multinomial algorithm is correct", {
  set.seed(1)
  prob <- runif(10)
  prob <- prob / sum(prob)
  size <- 20
  n <- 5

  res <- dust_rng$new(1, seed = 1L)$multinomial(n, size, prob)

  ## Separate implementation of the core algorithm:
  cmp_multinomial <- function(rng, size, prob) {
    p <- prob / (1 - cumsum(c(0, prob[-length(prob)])))
    ret <- numeric(length(prob))
    for (i in seq_len(length(prob) - 1L)) {
      ret[i] <- rng$binomial(1, size, p[i])
      size <- size - ret[i]
    }
    ret[length(ret)] <- size
    ret
  }

  rng2 <- dust_rng$new(1, seed = 1L)
  cmp <- replicate(n, cmp_multinomial(rng2, size, prob))
  expect_equal(res, cmp)
})


test_that("multinomial expectation is correct", {
  p <- runif(10)
  p <- p / sum(p)
  n <- 10000
  res <- dust_rng$new(1, seed = 1L)$multinomial(n, 100, p)
  expect_equal(dim(res), c(10, n))
  expect_equal(colSums(res), rep(100, n))
  expect_equal(rowMeans(res), p * 100, tolerance = 1e-2)
})


test_that("multinomial allows zero probs", {
  p <- runif(10)
  p[4] <- 0
  p <- p / sum(p)
  n <- 500
  size <- 100
  res <- dust_rng$new(1, seed = 1L)$multinomial(n, size, p)

  expect_equal(res[4, ], rep(0, n))
  expect_equal(colSums(res), rep(size, n))
})


test_that("multinomial allows non-normalised prob", {
  p <- runif(10, 0, 10)
  n <- 50
  res1 <- dust_rng$new(1, seed = 1L)$multinomial(n, 100, p)
  res2 <- dust_rng$new(1, seed = 1L)$multinomial(n, 100, p / sum(p))
  expect_equal(res1, res2)
})


test_that("Invalid prob throws an error", {
  r <- dust_rng$new(1, seed = 1L)
  expect_error(
    r$multinomial(1, 10, c(0, 0, 0)),
    "No positive prob in call to multinomial")
  expect_error(
    r$multinomial(1, 10, c(-0.1, 0.6, 0.5)),
    "Negative prob passed to multinomial")
})


test_that("Can vary parameters for multinomial, single generator", {
  np <- 7L
  ng <- 1L
  size <- 13
  n <- 17L
  prob <- matrix(runif(np * n), np, n)
  prob <- prob / rep(colSums(prob), each = np)

  rng <- dust_rng$new(1, seed = 1L)
  cmp <- vapply(seq_len(n), function(i) rng$multinomial(1, size, prob[, i]),
                numeric(np))
  res <- dust_rng$new(1, seed = 1L)$multinomial(n, size, prob)
  expect_equal(res, cmp)

  expect_error(
    dust_rng$new(1, seed = 1L)$multinomial(n, size, prob[, -5]),
    "If 'prob' is a matrix, it must have 17 columns")
  expect_error(
    dust_rng$new(1, seed = 1L)$multinomial(n, size, prob[0, ]),
    "Input parameters imply length of 'prob' of only 0 (< 2)",
    fixed = TRUE)
  expect_error(
    dust_rng$new(1, seed = 1L)$multinomial(n, size, prob[1, , drop = FALSE]),
    "Input parameters imply length of 'prob' of only 1 (< 2)",
    fixed = TRUE)
})


test_that("Can vary parameters by generator for multinomial", {
  np <- 7L
  ng <- 3L
  size <- 13
  n <- 17L

  prob <- array(runif(np * ng), c(np, 1, ng))
  prob <- prob / rep(colSums(prob), each = np)

  state <- matrix(dust_rng$new(ng, seed = 1L)$state(), ncol = ng)
  cmp <- vapply(seq_len(ng), function(i)
    dust_rng$new(1, seed = state[, i])$multinomial(n, size, prob[, , i]),
    matrix(numeric(), np, n))

  res <- dust_rng$new(ng, seed = 1L)$multinomial(n, size, prob)
  expect_equal(res, cmp)
})


test_that("Can vary parameters for multinomial, multiple generators", {
  np <- 7L
  ng <- 3L
  size <- 13
  n <- 17L
  prob <- array(runif(np * n * ng), c(np, n, ng))
  prob <- prob / rep(colSums(prob), each = np)

  ## Setting up the expectation here is not easy, we need a set of
  ## generators. This test exploits the fact that we alredy worked out
  ## we could vary a parameter over draws with a single generator.
  state <- matrix(dust_rng$new(ng, seed = 1L)$state(), ncol = ng)
  cmp <- vapply(seq_len(ng), function(i)
    dust_rng$new(1, seed = state[, i])$multinomial(n, size, prob[, , i]),
    matrix(numeric(), np, n))

  res <- dust_rng$new(ng, seed = 1L)$multinomial(n, size, prob)
  expect_equal(res, cmp)

  expect_error(
    dust_rng$new(ng, seed = 1L)$multinomial(n, size, prob[, -5, ]),
    "If 'prob' is a 3d array, it must have 1 or 17 columns")
  expect_error(
    dust_rng$new(ng, seed = 1L)$multinomial(n, size, prob[, , -1]),
    "If 'prob' is a 3d array, it must have 3 layers")
  expect_error(
    dust_rng$new(ng, seed = 1L)$multinomial(n, size, prob[0, , ]),
    "Input parameters imply length of 'prob' of only 0 (< 2)",
    fixed = TRUE)
  ## Final bad inputs:
  p4 <- array(prob, c(dim(prob), 1))
  expect_error(
    dust_rng$new(ng, seed = 1L)$multinomial(n, size, p4),
    "'prob' must be a vector, matrix or 3d array")
})


test_that("multinomial random numbers from floats have correct distribution", {
  n <- 100000
  prob <- runif(7)
  prob <- prob / sum(prob)
  size <- 13
  yf <- dust_rng$new(1, real_type = "float")$multinomial(n, size, prob)
  expect_equal(dim(yf), c(length(prob), n))
  expect_equal(colSums(yf), rep(size, n))
  expect_equal(rowMeans(yf), size * prob,
               tolerance = 1e-2)
})


test_that("long jump", {
  seed <- 1
  rng1 <- dust_rng$new(seed, real_type = "float")
  rng2 <- dust_rng$new(seed, real_type = "float")$jump()
  rng3 <- dust_rng$new(seed, real_type = "float")$long_jump()
  rng4 <- dust_rng$new(seed, real_type = "float")$long_jump()$jump()

  r1 <- rng1$random_real(20)
  r2 <- rng2$random_real(20)
  r3 <- rng3$random_real(20)
  r4 <- rng4$random_real(20)

  expect_true(all(r1 != r2))
  expect_true(all(r1 != r3))
  expect_true(all(r1 != r4))
  expect_true(all(r2 != r3))
  expect_true(all(r2 != r4))
  expect_true(all(r3 != r4))
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

  rng_f <- dust_rng$new(1, real_type = "float", deterministic = TRUE)
  rng_d <- dust_rng$new(1, real_type = "double", deterministic = TRUE)
  state_f <- rng_f$state()
  state_d <- rng_d$state()

  expect_equal(rng_f$binomial(m, n, p), n * p, tolerance = 1e-6)
  expect_equal(rng_d$binomial(m, n, p), n * p)

  expect_equal(rng_f$state(), state_f)
  expect_equal(rng_d$state(), state_d)
})


test_that("deterministic rbinom accepts non-integer size", {
  m <- 10
  n <- runif(m, 0, 10)
  p <- runif(m)
  rng_f <- dust_rng$new(1, real_type = "float", deterministic = TRUE)
  rng_d <- dust_rng$new(1, real_type = "double", deterministic = TRUE)
  state_f <- rng_f$state()
  state_d <- rng_d$state()
  expect_equal(rng_f$binomial(m, n, p), n * p, tolerance = 1e-6)
  expect_equal(rng_d$binomial(m, n, p), n * p)
  expect_equal(rng_f$state(), state_f)
  expect_equal(rng_d$state(), state_d)
})


test_that("deterministic rbinom allow small negative innacuracies", {
  m <- 10
  n <- runif(m, 0, 10)
  p <- runif(m)
  rng_f <- dust_rng$new(1, real_type = "float", deterministic = TRUE)
  rng_d <- dust_rng$new(1, real_type = "double", deterministic = TRUE)

  eps_d <- .Machine$double.eps
  eps_f <- 2^-23

  expect_identical(rng_f$binomial(1, 0, 0.5), 0.0)
  expect_identical(rng_d$binomial(1, 0, 0.5), 0.0)
  expect_identical(rng_f$binomial(1, -eps_f, 0.5), 0.0)
  expect_identical(rng_d$binomial(1, -eps_d, 0.5), 0.0)

  expect_error(rng_f$binomial(1, -sqrt(eps_f * 1.1), 0.5),
               "Invalid call to binomial with n = -")
  expect_error(rng_d$binomial(1, -sqrt(eps_d * 1.1), 0.5),
               "Invalid call to binomial with n = -")
})


test_that("deterministic rpois returns mean", {
  m <- 10
  lambda <- runif(m, 0, 50)
  rng_f <- dust_rng$new(1, real_type = "float", deterministic = TRUE)
  rng_d <- dust_rng$new(1, real_type = "double", deterministic = TRUE)
  state_f <- rng_f$state()
  state_d <- rng_d$state()
  expect_equal(rng_f$poisson(m, lambda), lambda, tolerance = 1e-6)
  expect_equal(rng_d$poisson(m, lambda), lambda)
  expect_equal(rng_f$state(), state_f)
  expect_equal(rng_d$state(), state_d)
})


test_that("deterministic rpois returns mean", {
  m <- 10
  lambda <- runif(m, 0, 50)
  rng_f <- dust_rng$new(1, real_type = "float", deterministic = TRUE)
  rng_d <- dust_rng$new(1, real_type = "double", deterministic = TRUE)
  state_f <- rng_f$state()
  state_d <- rng_d$state()
  expect_equal(rng_f$poisson(m, lambda), lambda, tolerance = 1e-6)
  expect_equal(rng_d$poisson(m, lambda), lambda)
  expect_equal(rng_f$state(), state_f)
  expect_equal(rng_d$state(), state_d)
})


test_that("deterministic runif returns mean", {
  m <- 10
  l <- runif(m, -10, 10)
  u <- l + runif(m, 0, 10)
  rng_f <- dust_rng$new(1, real_type = "float", deterministic = TRUE)
  rng_d <- dust_rng$new(1, real_type = "double", deterministic = TRUE)
  state_f <- rng_f$state()
  state_d <- rng_d$state()
  expect_equal(rng_f$uniform(m, l, u), (l + u) / 2, tolerance = 1e-6)
  expect_equal(rng_d$uniform(m, l, u), (l + u) / 2)
  expect_equal(rng_f$state(), state_f)
  expect_equal(rng_d$state(), state_d)
})


test_that("deterministic rexp returns mean", {
  m <- 10
  rate <- runif(m, 0, 10)
  rng_f <- dust_rng$new(1, real_type = "float", deterministic = TRUE)
  rng_d <- dust_rng$new(1, real_type = "double", deterministic = TRUE)
  state_f <- rng_f$state()
  state_d <- rng_d$state()
  expect_equal(rng_f$exponential(m, rate), 1 / rate, tolerance = 1e-6)
  expect_equal(rng_d$exponential(m, rate), 1 / rate)
  expect_equal(rng_f$state(), state_f)
  expect_equal(rng_d$state(), state_d)
})


test_that("deterministic rnorm returns mean", {
  m <- 10
  mu <- runif(m, -10, 10)
  sd <- runif(m, 0, 10)
  rng_f <- dust_rng$new(1, real_type = "float", deterministic = TRUE)
  rng_d <- dust_rng$new(1, real_type = "double", deterministic = TRUE)
  state_f <- rng_f$state()
  state_d <- rng_d$state()
  expect_equal(rng_f$normal(m, mu, sd), mu, tolerance = 1e-6)
  expect_equal(rng_d$normal(m, mu, sd), mu)
  expect_equal(rng_f$state(), state_f)
  expect_equal(rng_d$state(), state_d)
})


test_that("Parameter expansion", {
  rng <- dust_rng$new(1, 10)

  m <- matrix(as.numeric(1:30), 3, 10)
  rng <- dust_rng$new(1, 10)
  expect_equal(floor(rng$uniform(3, m, m + 1)), m)

  expect_equal(floor(rng$uniform(3, m[, 1], m[, 1] + 1)),
               matrix(as.numeric(1:3), 3, 10))
  expect_equal(floor(rng$uniform(3, 1, 2)),
               matrix(1, 3, 10))
  m1 <- m[1, , drop = FALSE]
  expect_equal(floor(rng$uniform(3, m1, m1 + 1)),
               m1[c(1, 1, 1), ])

  expect_error(
    rng$uniform(3, c(1, 2, 3, 4), 10),
    "If 'min' is a vector, it must have 1 or 3 elements")
  expect_error(
    rng$uniform(3, m[, 1:2], 10),
    "If 'min' is a matrix, it must have 10 columns")
  expect_error(
    rng$uniform(3, m[1:2, ], 10),
    "If 'min' is a matrix, it must have 1 or 3 rows")
})


test_that("We can load the example rng package", {
  skip_for_compilation()
  skip_on_os("windows")

  path_src <- dust_file("random/package")
  tmp <- tempfile()
  copy_directory(path_src, tmp)
  cpp11::cpp_register(tmp, quiet = TRUE)

  pkg <- pkgload::load_all(tmp, export_all = FALSE, quiet = TRUE)
  ans <- pkg$env$random_normal(10, 0, 1, 42)
  cmp <- dust_rng$new(42)$normal(10, 0, 1)
  expect_equal(ans, cmp)

  pkgload::unload("rnguse")
  unlink(tmp, recursive = TRUE)
})


test_that("We can compile the standalone program", {
  skip_for_compilation()
  skip_on_os("windows")

  path_src <- dust_file("random/openmp")
  tmp <- tempfile()
  copy_directory(path_src, tmp)

  args <- c(dirname(dust_file("include")), "--no-openmp")

  code <- withr::with_dir(
    tmp,
    system2("./configure", args, stdout = FALSE, stderr = FALSE))
  expect_equal(code, 0)
  code <- withr::with_dir(
    tmp,
    system2("make", stdout = FALSE, stderr = FALSE))
  expect_equal(code, 0)

  res <- system2(file.path(tmp, "rnguse"), c("10", "5", "42"),
                 stdout = TRUE)
  ans <- as.numeric(sub("[0-9]: ", "", res))

  cmp <- colSums(dust_rng$new(42, 5)$uniform(10, 0, 1))
  expect_equal(ans, cmp)
})
