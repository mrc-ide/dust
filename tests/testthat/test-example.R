test_that("create walk, stepping for one step", {
  res <- dust_example("walk")
  obj <- res$new(list(sd = 1), 0, 10, seed = 1L)
  expect_null(obj$info())

  y <- obj$run(1)
  cmp <- dust_rng$new(1, 10)$rnorm(10, 0, 1)
  expect_identical(drop(y), cmp)
})


test_that("walk agrees with random number stream", {
  res <- dust_example("walk")
  obj <- res$new(list(sd = 1), 0, 10, seed = 1L)
  y <- obj$run(5)

  cmp <- dust_rng$new(1, 10)$rnorm(50, 0, 1)
  expect_equal(drop(y), colSums(matrix(cmp, 5, 10, byrow = TRUE)))
  expect_identical(obj$state(), y)
})


test_that("Reset particles and resume continues with rng", {
  res <- dust_example("walk")
  sd1 <- 2
  sd2 <- 4

  obj <- res$new(list(sd = sd1), 0, 10, seed = 1L)

  y1 <- obj$run(5)
  expect_equal(obj$step(), 5)

  obj$reset(list(sd = sd2), 0)
  ## obj$update_state(list(sd = sd2), NULL, 0, set_initial_state = TRUE)

  expect_equal(obj$step(), 0)
  y2 <- obj$run(5)
  expect_equal(obj$step(), 5)

  ## Then draw the random numbers:
  cmp <- dust_rng$new(1, 10)$rnorm(100, 0, 1)
  m1 <- matrix(cmp[1:50], 5, 10, byrow = TRUE)
  m2 <- matrix(cmp[51:100], 5, 10, byrow = TRUE)

  expect_equal(drop(y1), colSums(m1) * sd1)
  expect_equal(drop(y2), colSums(m2) * sd2)
})


test_that("Basic sir model", {
  res <- dust_example("sir")

  obj <- res$new(list(), 0, 100, seed = 1L)
  obj$set_index(1L)

  ans <- vector("list", 150)
  for (i in seq_along(ans)) {
    value <- obj$run(i * 4)
    ans[[i]] <- list(value = value, state = obj$state(), step = obj$step())
  }

  step <- vapply(ans, function(x) x$step, numeric(1))
  state_s <- t(vapply(ans, function(x) x$state[1, ], numeric(100)))
  state_i <- t(vapply(ans, function(x) x$state[2, ], numeric(100)))
  state_r <- t(vapply(ans, function(x) x$state[3, ], numeric(100)))
  state_cum <- t(vapply(ans, function(x) x$state[4, ], numeric(100)))
  state_inc <- t(vapply(ans, function(x) x$state[5, ], numeric(100)))
  value <- t(vapply(ans, function(x) drop(x$value), numeric(100)))

  n <- nrow(state_s)
  expect_true(all(state_s[-n, ] - state_s[-1, ] >= 0))
  expect_true(all(state_r[-n, ] - state_r[-1, ] <= 0))
  expect_false(all(state_i[-n, ] - state_i[-1, ] <= 0))
  expect_false(all(state_i[-n, ] - state_i[-1, ] >= 0))
  expect_identical(value, state_s)
  expect_equal(step, seq(4, by = 4, length.out = n))

  expect_equal(state_cum, 1000 - state_s)
  expect_equal(state_inc[-1, ], diff(state_cum))

  s <- ans[[150]]$state
  expect_equal(obj$state(), s)
  expect_equal(obj$state(1L), s[1, , drop = FALSE])
  expect_equal(obj$state(3:1), s[3:1, , drop = FALSE])
  expect_equal(obj$state(c(2L, 2L)), s[c(2, 2), , drop = FALSE])
  expect_error(obj$state(0L),
               "All elements of 'index' must lie in [1, 5]",
               fixed = TRUE)
  expect_error(obj$state(1:6),
               "All elements of 'index' must lie in [1, 5]",
               fixed = TRUE)
})


test_that("set index", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 1, seed = 1L)
  expect_equal(mod$state(), matrix(1:10))
  expect_equal(mod$run(0), matrix(1:10, nrow = 10, ncol = 1))

  mod$set_index(2:4)
  expect_equal(mod$run(0), matrix(2:4))

  y <- mod$run(1)
  expect_equal(y, mod$state(2:4))
  expect_equal(y, mod$state()[2:4, , drop = FALSE])

  mod$set_index(integer(0))
  expect_equal(mod$run(1), matrix(numeric(0), nrow = 0, ncol = 1))
})


test_that("reset does not clear the index", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 1, seed = 1L)
  mod$set_index(2:4)
  expect_equal(mod$run(0), matrix(2:4))
  mod$reset(list(len = 10), 0)
  expect_equal(mod$run(0), matrix(2:4))
})


test_that("set model state", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 1, seed = 1L)
  expect_equal(mod$state(), matrix(1:10))
  x <- runif(10)
  mod$update_state(state = x)
  expect_equal(mod$state(), matrix(x))
  expect_error(
    mod$update_state(state = 1),
    "Expected a vector of length 10 for 'state'")
  expect_equal(mod$state(), matrix(x))
})


test_that("set model state into multiple particles", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 20, seed = 1L)
  expect_equal(mod$state(), matrix(1:10, 10, 20))
  x <- runif(10)
  mod$update_state(state = x)
  expect_equal(mod$state(), matrix(x, 10, 20))
  expect_error(
    mod$update_state(state = 1),
    "Expected a vector of length 10 for 'state'")
  expect_equal(mod$state(), matrix(x, 10, 20))
})


test_that("set model state with a matrix", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 2, seed = 1L)
  m <- matrix(runif(20), 10, 2)
  mod$update_state(state = m)

  expect_equal(mod$state(), m)
  expect_error(
    mod$update_state(state = m[, c(1, 1, 2, 2)]),
    "Expected a matrix with 2 cols for 'state' but given 4")
  expect_error(
    mod$update_state(state = m[1:5, ]),
    "Expected a matrix with 10 rows for 'state' but given 5")
  expect_equal(mod$state(), m)
})


test_that("reorder", {
  res <- dust_example("walk")

  obj <- res$new(list(sd = 1), 0, 10, seed = 1L)
  y1 <- obj$run(5)

  ## Simplest permutation:
  index <- rev(seq_along(y1))

  obj$reorder(index)
  y2 <- obj$state()
  expect_equal(drop(y2), rev(y1))

  y3 <- obj$run(10)

  cmp <- dust_rng$new(1, 10)$rnorm(100, 0, 1)
  m1 <- matrix(cmp[1:50], 5, 10, byrow = TRUE)
  m2 <- m1[, index]
  m3 <- matrix(cmp[51:100], 5, 10, byrow = TRUE)

  expect_equal(drop(y1), colSums(m1))
  expect_equal(drop(y2), colSums(m2))
  expect_equal(drop(y3), colSums(rbind(m2, m3)))
})


test_that("reorder and duplicate", {
  res <- dust_example("walk")

  obj <- res$new(list(sd = 1), 0, 10, seed = 1L)
  y1 <- obj$run(5)

  index <- c(1L, 4L, 9L, 7L, 7L, 2L, 5L, 9L, 9L, 5L)

  obj$reorder(index)
  y2 <- obj$state()
  expect_equal(drop(y2), y1[index])

  y3 <- obj$run(10)

  cmp <- dust_rng$new(1, 10)$rnorm(100, 0, 1)
  m1 <- matrix(cmp[1:50], 5, 10, byrow = TRUE)
  m2 <- m1[, index]
  m3 <- matrix(cmp[51:100], 5, 10, byrow = TRUE)

  expect_equal(drop(y1), colSums(m1))
  expect_equal(drop(y2), colSums(m2))
  expect_equal(drop(y3), colSums(rbind(m2, m3)))
})


test_that("validate reorder vector is correct length", {
  res <- dust_example("walk")
  obj <- res$new(list(sd = 1), 0, 10, seed = 1L)
  expect_error(obj$reorder(1L),
               "Expected a vector of length 10 for 'index'")
  expect_error(obj$reorder(rep(1L, 100)),
               "Expected a vector of length 10 for 'index'")
  expect_error(obj$reorder(cbind(1:10)),
               "Expected a vector for 'index'")
})


test_that("validate reorder vector is in correct range", {
  res <- dust_example("walk")
  obj <- res$new(list(sd = 1), 0, 10, seed = 1L)
  index <- seq_len(10)
  msg <- "All elements of 'index' must lie in [1, 10]"

  index[5] <- 0L
  expect_error(obj$reorder(index), msg, fixed = TRUE)

  index[5] <- 11
  expect_error(obj$reorder(index), msg, fixed = TRUE)

  index[5] <- -1L
  expect_error(obj$reorder(index), msg, fixed = TRUE)
})


test_that("run in float mode", {
  skip_for_compilation()
  res_d <- dust_example("walk")
  res_f <- dust(dust_file("examples/walk.cpp"), real_t = "float", quiet = TRUE)

  n <- 1000
  obj_d <- res_d$new(list(sd = 10), 0, n, seed = 1L)
  obj_f <- res_f$new(list(sd = 10), 0, n, seed = 1L)

  expect_equal(obj_d$device_info()$real_bits, 64)
  expect_equal(obj_f$device_info()$real_bits, 32)

  y_d <- obj_d$run(10)
  y_f <- obj_f$run(10)

  expect_equal(y_d, y_f, tolerance = 1e-5)
  expect_false(identical(y_d, y_f))
})


test_that("reset changes info", {
  res <- dust_example("sir")
  obj <- res$new(list(), 0, 100, seed = 1L)
  expect_equal(obj$info(),
               list(vars = c("S", "I", "R", "cases_cumul", "cases_inc"),
                    pars = list(beta = 0.2, gamma = 0.1)))
  obj$reset(list(beta = 0.1), 0)
  expect_equal(obj$info(),
               list(vars = c("S", "I", "R", "cases_cumul", "cases_inc"),
                    pars = list(beta = 0.1, gamma = 0.1)))
})


test_that("Basic threading test", {
  skip_for_compilation()
  res <- dust(dust_file("examples/parallel.cpp"), quiet = TRUE)

  obj <- res$new(list(sd = 1), 0, 10, n_threads = 2L, seed = 1L)
  obj$set_index(1L)
  y0 <- obj$state()
  y22_1 <- obj$run(5)
  y22_2 <- obj$state()

  ## And again without parallel
  obj <- res$new(list(sd = 1), 0, 10, n_threads = 1L, seed = 1L)
  obj$set_index(1L)
  y12_1 <- obj$run(5)
  y12_2 <- obj$state()

  ## Quick easy check:
  expect_equal(y22_1[1, ], y22_2[1, ])
  expect_equal(y12_1[1, ], y12_2[1, ])

  if (obj$has_openmp() && y0[2, 1] == 1) {
    ## OMP is enabled (note: this relies on implementation details of
    ## openmp and we'd need to change this for a CRAN release - what
    ## we'd expect to see is that 0 and 1 are both present, but as
    ## we're leaving it up to the compiler to set the schedule I
    ## believe this is not reliable theoretically, even though it is
    ## empirically)
    expect_equal(y22_2[2, ], rep(0:1, each = 5))
  } else {
    expect_equal(y22_2[2, ], rep(-1, 10))
  }

  expect_equal(y22_1, y12_1)
})


## It's quite hard to test the cache well; we might wrap it up a bit
## nicer than it is. The trick is we can't safely unload things
## (because the finalisers might run) and we can't easily purge the
## cache and show that it *would* compile either. And we can't just
## compile and load the same thing twice because the names of the dlls
## will clash.
##
## This at least shows that on the second time round we don't compile
## and we get the right data out.
test_that("cache hits do not compile", {
  skip_on_cran()
  cmp <- dust(dust_file("examples/walk.cpp"), quiet = TRUE)
  expect_message(
    res <- dust(dust_file("examples/walk.cpp"), quiet = FALSE),
    "Using cached model")
  ## Same object
  expect_identical(res, cmp)
})


test_that("set model state and time, varying time", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 2, seed = 1L)
  m <- matrix(rep(as.numeric(1:2), each = 10), 10, 2)
  step <- 0:1
  mod$update_state(state = m, step = step)
  cmp <- dust_rng$new(1, 1)$rnorm(10, 0, 1)

  state <- mod$state()
  expect_equal(mod$step(), 1)
  expect_equal(state[, 2], m[, 2])
  expect_equal(state[, 1], m[, 1] + cmp)
})


test_that("setting model state and step requires correct length step", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 5, seed = 1L)
  m <- matrix(rep(as.numeric(1:5), each = 10), 10, 5)
  expect_error(
    mod$update_state(state = m, step = 0:3),
    "Expected 'step' to be scalar or length 5")
  expect_error(
    mod$update_state(state = m, step = 0:7),
    "Expected 'step' to be scalar or length 5")
})


test_that("set model state and time, constant time", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 2, seed = 1L)
  m <- matrix(runif(20), 10, 2)
  step <- 10L
  mod$update_state(state = m, step = step)

  state <- mod$state()
  expect_equal(mod$step(), 10)
  expect_equal(state, m)
})


test_that("set model time but not state", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 2, seed = 1L)
  expect_null(mod$update_state(state = NULL, step = NULL))
  expect_equal(mod$step(), 0)
  expect_equal(mod$state(), matrix(1:10, 10, 2))

  expect_null(mod$update_state(state = NULL, step = 10L))
  expect_equal(mod$step(), 10)
  expect_equal(mod$state(), matrix(1:10, 10, 2))
})


test_that("NULL state leaves state untouched", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 2, seed = 1L)
  m <- matrix(runif(20), 10, 2)
  mod$update_state(state = m, step = NULL)
  expect_equal(mod$state(), m)
  expect_equal(mod$step(), 0)

  mod$update_state(state = NULL, step = 10L)
  expect_equal(mod$state(), m)
  expect_equal(mod$step(), 10)

  mod$update_state(state = NULL, step = NULL)
  expect_equal(mod$state(), m)
  expect_equal(mod$step(), 10)
})


test_that("type coersion in setting index", {
  res <- dust_example("sir")
  obj <- res$new(list(), 0, 100, seed = 1L)
  expect_null(obj$set_index(1L))
  expect_null(obj$set_index(1))
  expect_error(
    obj$set_index(1.5),
    "All elements of 'index' must be integer-like",
    fixed = TRUE)
  expect_error(
    obj$set_index(TRUE),
    "Expected a numeric vector for 'index'",
    fixed = TRUE)
})


test_that("can't load invalid example", {
  expect_error(dust_example("asdf"), "Unknown example 'asdf'")
})


test_that("can run volatility example", {
  res <- dust_example("volatility")
  obj <- res$new(list(), 0, 5000, seed = 1L)
  y <- drop(obj$simulate(0:100))
  expect_lt(diff(range(colMeans(y))), 0.5)
})


test_that("has_openmp can be called statically and normally", {
  res <- dust_example("volatility")
  expected <- openmp_info()$has_openmp
  expect_equal(res$public_methods$has_openmp(), expected)
  expect_equal(res$new(list(), 0, 1)$has_openmp(), expected)
})


test_that("Sensible behaviour of compare_data if not implemented", {
  res <- dust_example("walk")
  expect_false(res$public_methods$has_compare())
  mod <- res$new(list(sd = 1), 0, 1, seed = 1L)
  expect_false(mod$has_compare())
  expect_error(
    mod$set_data(list(list(1, list()))),
    "The 'set_data' method is not supported for this class")
  expect_error(
    mod$set_data(list()),
    "The 'set_data' method is not supported for this class")
  expect_error(
    mod$compare_data(),
    "The 'compare_data' method is not supported for this class")
  expect_error(
    mod$filter(),
    "The 'filter' method is not supported for this class")
})


test_that("Can run compare_data", {
  res <- dust_example("sir")
  expect_true(res$public_methods$has_compare())

  np <- 10
  end <- 150 * 4
  steps <- seq(0, end, by = 4)
  ans <- res$new(list(), 0, np, seed = 1L)$simulate(steps)

  ## Confirm the incidence calculation is correct:
  expect_equal(
    rowSums(ans[5, , ]),
    1000 - ans[1, , length(steps)])
  expect_equal(
    t(apply(ans[5, , ], 1, cumsum)),
    ans[4, , ])

  d <- dust_data(data.frame(step = steps, incidence = ans[5, 1, ]))

  ## Use Inf for exp_noise as that gives us deterministic results
  mod <- res$new(list(exp_noise = Inf), 0, np, seed = 1L)
  expect_true(mod$has_compare())
  mod$run(36)
  expect_null(mod$compare_data())

  mod$set_data(d)
  x <- mod$compare_data()
  expect_length(x, np)

  ## Need to compute manually our probability and check against the
  ## computed version:
  modelled <- drop(mod$state(5))
  expect_equal(dpois(d[[10]][[2]]$incidence, modelled, log = TRUE), x)
})


test_that("fetch model size", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 7, seed = 1L)
  expect_equal(mod$n_particles(), 7)
  expect_equal(mod$n_state(), 10)
  mod$set_index(c(3, 5, 7))
  expect_equal(mod$n_state(), 10)
})


test_that("resample", {
  res <- dust_example("variable")
  obj <- res$new(list(len = 5), 0, 7, seed = 1L)
  m <- matrix(as.numeric(1:35), 5, 7)
  obj$update_state(state = m)
  rng <- dust_rng$new(obj$rng_state(last_only = TRUE))
  u <- rng$unif_rand(1)
  w <- runif(obj$n_particles())

  idx <- obj$resample(w)
  expect_equal(idx, resample_index(w, u))
  expect_equal(obj$state(), m[, idx])
})


test_that("resample error cases", {
  res <- dust_example("variable")
  obj <- res$new(list(len = 5), 0, 7, seed = 1L)
  m <- matrix(as.numeric(1:35), 5, 7)
  obj$update_state(state = m)
  rng <- dust_rng$new(obj$rng_state(last_only = TRUE))
  u <- rng$unif_rand(1)
  w <- runif(obj$n_particles())

  expect_error(obj$resample(w[-1]),
               "Expected a vector of length 7 for 'weights'")
  expect_error(obj$resample(c(w[-1], -1)),
               "All weights must be positive")
  expect_identical(obj$state(), m)

  ## This corner case is pretty nasty, and practically the particle
  ## filter will have stopped by this point.
  idx <- obj$resample(rep(0, 7))
  expect_equal(idx, rep(1, 7))
})


test_that("volality compare is correct", {
  dat <- example_volatility()
  pars <- dat$pars

  np <- 400L
  mod <- volatility$new(pars, 0, np, seed = 1L)
  expect_null(mod$compare_data())
  mod$set_data(dust_data(dat$data))

  y <- mod$run(1)
  expect_equal(mod$compare_data(),
               dat$compare(y, dat$data[1, ], pars))

  f <- function() {
    mod$reset(pars, 0L)
    mod$filter()$log_likelihood
  }
  ll <- replicate(200, f())

  ll_true <- dat$kalman_filter(pars, dat$data)
  expect_lt(min(ll), ll_true)
  expect_gt(max(ll), ll_true)
  expect_equal(mean(ll), ll_true, tolerance = 0.01)
})


test_that("can simulate, respecting index", {
  res <- dust_example("sir")

  steps <- seq(0, 100, by = 10)
  np <- 20
  mod <- res$new(list(), 0, np, seed = 1L)
  y <- mod$simulate(steps)
  expect_equal(dim(y), c(5, np, length(steps)))

  mod2 <- res$new(list(), 0, np, seed = 1L)
  mod2$set_index(5)
  expect_identical(mod2$simulate(steps), y[5, , , drop = FALSE])
})


test_that("validate simulate steps", {
  res <- dust_example("sir")

  np <- 20
  mod <- res$new(list(len = 5), 10, np, seed = 1L)
  y <- matrix(runif(np * 5), 5, np)
  mod$update_state(state = y)

  expect_error(
    mod$simulate(integer(0)),
    "'step_end' must have at least one element")
  expect_error(
    mod$simulate(0:10),
    "'step_end[1]' must be at least 10", fixed = TRUE)
  expect_error(
    mod$simulate(10:5),
    "'step_end' must be non-decreasing (error on element 2)", fixed = TRUE)
  expect_error(
    mod$simulate(c(10, 20, 30, 20, 50)),
    "'step_end' must be non-decreasing (error on element 4)", fixed = TRUE)

  ## Unchanged
  expect_equal(mod$step(), 10)
  expect_equal(mod$state(), y)
})


test_that("no device info by default", {
  no_cuda <- list(has_cuda = FALSE,
                  cuda_version = NULL,
                  devices = data.frame(id = integer(0),
                                       name = character(0),
                                       memory = numeric(0),
                                       version = integer(0),
                                       stringsAsFactors = FALSE),
                  real_bits = 64L)
  res <- dust_example("sir")
  expect_false(res$public_methods$has_cuda())
  expect_equal(res$public_methods$device_info(), no_cuda)

  mod <- res$new(list(), 0, 1)
  expect_false(mod$has_cuda())
  expect_equal(mod$device_info(), no_cuda)
})


test_that("throw when triggering invalid binomials", {
  res <- dust_example("sir")
  mod <- res$new(list(), 0, 10)
  s <- mod$state()
  s[2, c(4, 9)] <- c(-1, -10)
  mod$update_state(state = s)

  err <- expect_error(
    mod$run(10),
    "2 particles reported errors")
  expect_match(
    err$message,
    "- 4: Invalid call to rbinom with n = -1, p =")
  expect_match(
    err$message,
    "- 9: Invalid call to rbinom with n = -10, p =")
  expect_equal(mod$state()[, c(4, 9)], s[, c(4, 9)])
  expect_error(mod$step(), "Errors pending; reset required")
  expect_error(mod$run(10), "Errors pending; reset required")
  expect_error(mod$simulate(10:20), "Errors pending; reset required")
  expect_error(mod$compare_data(), "Errors pending; reset required")
  expect_error(mod$filter(), "Errors pending; reset required")

  ## This will clear the errors:
  mod$update_state(state = abs(s), step = 0)
  ## And we can run again:
  expect_silent(mod$run(10))
  expect_equal(mod$step(), 10)
})


test_that("Truncate errors past certain point", {
  res <- dust_example("sir")
  mod <- res$new(list(), 0, 10)
  s <- mod$state()
  s[2, ] <- -10
  mod$update_state(state = s)
  err <- expect_error(
    mod$simulate(0:10),
    "10 particles reported errors")
  expect_match(
    err$message,
    "(and 6 more)", fixed = TRUE)
})


test_that("more binomial errors", {
  res <- dust_example("sir")
  set.seed(1)
  gamma <- runif(100, max = 0.1)
  gamma[c(10, 20, 30)] <- -gamma[c(10, 20, 30)]
  par <- apply(cbind(beta = 0.15, gamma = gamma), 1, as.list)
  mod <- res$new(par, 0, NULL, seed = 1L, pars_multi = TRUE)
  err <- expect_error(mod$run(10),
                      "3 particles reported errors")
  expect_match(err$message, "10: Invalid call to rbinom")
})


test_that("steps must not be negative", {
  res <- dust_example("sir")
  y0 <- matrix(1, 1, 5)
  pars <- rep(list(list(sd = 1)), 5)

  mod <- res$new(list(), 0, 1)
  expect_error(
    mod$simulate(c(0:10, -5)),
    "All elements of 'step_end' must be non-negative")
})


test_that("run random walk deterministically", {
  res <- dust_example("walk")
  obj <- res$new(list(sd = 1), 0, 100, seed = 1L)
  rng_state <- obj$rng_state()
  m <- obj$state()
  m[] <- runif(length(m))
  obj$update_state(state = m)
  expect_equal(obj$run(10, deterministic = TRUE), m)
  expect_equal(obj$rng_state(), rng_state)
})


test_that("run simulate deterministically", {
  res <- dust_example("walk")
  obj <- res$new(list(sd = 1), 0, 100, seed = 1L)
  rng_state <- obj$rng_state()
  m <- obj$state()
  m[] <- runif(length(m))
  obj$update_state(state = m)
  res <- obj$simulate(0:10, deterministic = TRUE)
  expect_equal(res,
               array(rep(m, 11), c(1, 100, 11)))
  expect_equal(obj$rng_state(), rng_state)
})


test_that("staggered start times, deterministically", {
  sir <- dust_example("sir")
  mod <- sir$new(list(), 0, 10, seed = 1L)

  state <- mod$state()
  n <- round(runif(10, -5, 5))
  state[1, ] <- state[1, ] - n
  state[2, ] <- state[2, ] + n
  step <- 1:10

  mod$update_state(state = state, step = step, deterministic = TRUE)

  res <- mod$state()
  f <- function(i) {
    mod <- sir$new(list(), 0, 1, seed = NULL)
    mod$update_state(state = state[, i], step = step[i])
    mod$run(10, deterministic = TRUE)
  }
  cmp <- vapply(1:10, f, numeric(5))
  expect_equal(res, cmp)
})
