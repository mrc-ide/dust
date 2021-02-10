context("multi")

test_that("create trivial multi dust object", {
  res <- dust_example("walk")
  obj1 <- res$new(list(sd = 1), 0, 10, seed = 1L, pars_multi = FALSE)
  obj2 <- res$new(list(list(sd = 1)), 0, 10, seed = 1L, pars_multi = TRUE)

  expect_identical(obj2$name(), obj1$name())
  expect_identical(obj2$param(), obj1$param())
  expect_identical(obj2$n_threads(), obj1$n_threads())
  expect_identical(obj2$has_openmp(), obj1$has_openmp())
  expect_identical(obj2$step(), obj1$step())

  expect_equal(obj2$pars(), list(obj1$pars()))
  expect_equal(obj2$info(), list(obj1$info()))

  expect_identical(obj2$rng_state(), obj1$rng_state())

  expect_identical(obj1$n_pars(), 0L)
  expect_identical(obj2$n_pars(), 1L)

  expect_identical(obj2$state(), array(obj1$state(), c(1, 10, 1)))
  expect_identical(obj2$state(1L), array(obj1$state(1L), c(1, 10, 1)))

  y1 <- obj1$run(1)
  y2 <- obj2$run(1)
})


test_that("create trivial 2 element mulitdust object", {
  res <- dust_example("walk")
  obj1 <- res$new(list(sd = 1), 0, 10, seed = 1L, pars_multi = FALSE)
  obj2 <- res$new(list(list(sd = 1), list(sd = 1)), 0, 5, seed = 1L,
                  pars_multi = TRUE)

  expect_identical(obj2$n_pars(), 2L)
  expect_equal(obj2$state(), array(obj1$state(), c(1, 5, 2)))

  y1 <- obj1$run(1)
  y2 <- obj2$run(1)
  expect_equal(y2, array(y1, c(1, 5, 2)))

  expect_identical(obj1$rng_state(), obj2$rng_state())
})


test_that("Can particles and resume continues with rng", {
  res <- dust_example("walk")
  sd1 <- 2
  sd2 <- 4
  sd3 <- 8

  pars1 <- list(list(sd = sd1), list(sd = sd2))
  pars2 <- list(list(sd = sd2), list(sd = sd3))

  nd <- length(pars1)
  np <- 10
  obj <- res$new(pars1, 0, np, seed = 1L, pars_multi = TRUE)

  ns <- 5
  y1 <- obj$run(ns)
  expect_equal(obj$step(), ns)
  obj$reset(pars2, 0)
  expect_equal(obj$step(), 0)
  y2 <- obj$run(ns)
  expect_equal(obj$step(), ns)

  ## Then draw the random numbers:
  rng <- dust_rng$new(1, 20)
  m1 <- array(rng$rnorm(ns * nd * np, 0, 1), c(np, nd, ns))
  m2 <- array(rng$rnorm(ns * nd * np, 0, 1), c(np, nd, ns))

  expect_equal(
    array(y1, c(np, nd)),
    apply(m1, 1:2, sum) * rep(c(sd1, sd2), each = np))

  expect_equal(
    array(y2, c(np, nd)),
    apply(m2, 1:2, sum) * rep(c(sd2, sd3), each = np))
})


test_that("Can set state", {
  res <- dust_example("variable")

  nd <- 3
  ns <- 7
  np <- 13

  pars <- rep(list(list(len = ns)), nd)
  mod <- res$new(pars, 0, np, seed = 1L, pars_multi = TRUE)

  ## Initial state:
  expect_equal(
    mod$state(),
    array(1:ns, c(ns, np, nd)))

  y <- mod$state()
  y[] <- runif(length(y))

  mod$set_state(y)
  expect_equal(mod$state(), y)

  expect_error(
    mod$set_state(c(y)),
    "Expected a 3d array for 'state' (but recieved a vector)",
    fixed = TRUE)
  expect_error(
    mod$set_state(matrix(c(y), c(ns, nd * np))),
    "Expected a 3d array for 'state'")
  expect_error(
    mod$set_state(y[-1, , ]),
    "Expected a 3d array with 7 rows for 'state'")
  expect_error(
    mod$set_state(y[, -1, ]),
    "Expected a 3d array with 13 columns for 'state'")
  expect_error(
    mod$set_state(y[, , -1]),
    "Expected a 3d array with dim[3] == 3 for 'state'",
    fixed = TRUE)

  ## Unchanged
  expect_equal(mod$state(), y)
})


test_that("can set index", {
  res <- dust_example("variable")
  nd <- 3
  ns <- 7
  np <- 13
  pars <- rep(list(list(len = ns)), nd)
  mod <- res$new(pars, 0, np, seed = 1L, pars_multi = TRUE)
  y <- mod$state()
  y[] <- runif(length(y))
  mod$set_state(y)
  idx <- c(1, 3, 5)
  mod$set_index(idx)

  expect_equal(mod$state(), y)
  expect_equal(mod$state(idx), y[idx, , ])
  expect_equal(mod$run(0), y[idx, , ])
})


test_that("can error if out-of-bounds index used", {
  res <- dust_example("variable")
  nd <- 3
  ns <- 7
  np <- 13
  pars <- rep(list(list(len = ns)), nd)
  mod <- res$new(pars, 0, np, seed = 1L, pars_multi = TRUE)
  expect_error(mod$set_index(c(1, 9, 2)),
               "All elements of 'index' must lie in [1, 7]",
               fixed = TRUE)
})


test_that("Can reorder particles (easy case)", {
  res <- dust_example("variable")
  nd <- 3
  ns <- 7
  np <- 4
  pars <- rep(list(list(len = ns)), nd)
  mod <- res$new(pars, 0, np, seed = 1L, pars_multi = TRUE)
  y <- array(as.numeric(seq_len(nd * ns * np)), c(ns, np, nd))
  mod$set_state(y)

  i <- cbind(c(1, 4, 2, 3),
             c(1, 2, 3, 4),
             c(1, 3, 4, 2))
  j <- i + rep((0:2) * 4, each = 4)
  z <- array(y, c(ns, np * nd))

  mod$reorder(i)
  ans <- mod$state()

  expect_equal(ans[, , 1], y[, i[, 1], 1])
  expect_equal(ans[, , 2], y[, i[, 2], 2])
  expect_equal(ans[, , 3], y[, i[, 3], 3])

  expect_equal(c(ans), c(z[, c(j)]))
})


test_that("Can reorder particles", {
  res <- dust_example("variable")
  nd <- 3
  ns <- 7
  np <- 13
  pars <- rep(list(list(len = ns)), nd)
  mod <- res$new(pars, 0, np, seed = 1L, pars_multi = TRUE)
  y <- array(as.numeric(seq_len(nd * ns * np)), c(ns, np, nd))
  mod$set_state(y)

  ## Our reorder matrix:
  i <- replicate(nd, sample.int(np, np, replace = TRUE))

  ## Compupting the reorder is actually hard
  cmp <- array(c(y[, i[, 1], 1],
                 y[, i[, 2], 2],
                 y[, i[, 3], 3]), dim(y))
  mod$reorder(i)
  expect_equal(mod$state(), cmp)
})


test_that("Can avoid invalid reorder index matrices", {
  res <- dust_example("variable")
  nd <- 3
  ns <- 7
  np <- 13
  pars <- rep(list(list(len = ns)), nd)
  mod <- res$new(pars, 0, np, seed = 1L, pars_multi = TRUE)
  y <- array(as.numeric(seq_len(nd * ns * np)), c(ns, np, nd))
  mod$set_state(y)

  ## Our reorder matrix:
  i <- replicate(nd, sample.int(np, np, replace = TRUE))
  expect_error(
    mod$reorder(c(i)),
    "Expected a matrix for 'index'")
  expect_error(
    mod$reorder(i[-1, ]),
    "Expected a matrix with 13 rows for 'index'")
  expect_error(
    mod$reorder(i[, -1]),
    "Expected a matrix with 3 columns for 'index'")
  i[1] <- np + 1
  expect_error(
    mod$reorder(i),
    "All elements of 'index' must lie in [1, 13]",
    fixed = TRUE)
  expect_equal(mod$state(), y)
})


test_that("set_pars is disabled", {
  res <- dust_example("variable")
  nd <- 3
  ns <- 7
  np <- 13
  pars <- rep(list(list(len = ns)), nd)
  mod <- res$new(pars, 0, np, seed = 1L, pars_multi = TRUE)
  expect_error(mod$set_pars(pars),
               "set_pars() with pars_multi not yet supported",
               fixed = TRUE)
})


test_that("must use same sized simulations", {
  res <- dust_example("variable")
  pars <- list(list(len = 7), list(len = 8))
  expect_error(
    res$new(pars, 0, 10, seed = 1L, pars_multi = TRUE),
    paste("Pars created different state sizes: pars 2 (of 2) had length 8",
          "but expected 7"),
    fixed = TRUE)
})


test_that("compare with multi pars", {
  res <- dust_example("sir2")

  np <- 10
  end <- 150 * 4
  steps <- seq(0, end, by = 4)
  ans <- dust_iterate(res$new(list(), 0, np, seed = 1L), steps)
  d <- data.frame(step = steps, incidence = ans[5, 1, ])

  ## Use Inf for exp_noise as that gives us deterministic results
  p <- list(exp_noise = Inf)
  mod <- res$new(rep(list(p), 3), 0, np, seed = 1L, pars_multi = TRUE)
  s <- mod$run(36)
  expect_null(mod$compare_data())

  mod$set_data(dust_data(d, multi = 3))
  x <- mod$compare_data()
  expect_equal(dim(x), c(10, 3))

  ## Then we try and replicate:
  cmp <- res$new(p, 0, np, seed = 1L)
  cmp$set_data(dust_data(d))
  cmp$run(36)
  cmp$set_state(s[, , 1])
  expect_equal(cmp$compare_data(), x[, 1])
  cmp$set_state(s[, , 2])
  expect_equal(cmp$compare_data(), x[, 2])
  cmp$set_state(s[, , 3])
  expect_equal(cmp$compare_data(), x[, 3])
})


test_that("compare with multi pars and different data", {
  res <- dust_example("sir2")

  np <- 10
  end <- 150 * 4
  steps <- seq(0, end, by = 4)
  ans <- dust_iterate(res$new(list(), 0, np, seed = 1L), steps)
  d <- data.frame(step = steps,
                  group = factor(rep(c("a", "b", "c"), each = length(steps))),
                  incidence = c(ans[5, 1, ], ans[5, 2, ], ans[5, 3, ]))

  ## Use Inf for exp_noise as that gives us deterministic results
  p <- list(exp_noise = Inf)
  mod <- res$new(rep(list(p), 3), 0, np, seed = 1L, pars_multi = TRUE)
  s <- mod$run(36)
  expect_null(mod$compare_data())

  mod$set_data(dust_data(d, multi = "group"))
  x <- mod$compare_data()
  expect_equal(dim(x), c(10, 3))

  ## Then we try and replicate:
  cmp <- res$new(p, 0, np, seed = 1L)
  cmp$run(36)

  cmp$set_data(dust_data(d[d$group == "a", ]))
  cmp$set_state(s[, , 1])
  expect_equal(cmp$compare_data(), x[, 1])

  cmp$set_data(dust_data(d[d$group == "b", ]))
  cmp$set_state(s[, , 2])
  expect_equal(cmp$compare_data(), x[, 2])

  cmp$set_data(dust_data(d[d$group == "c", ]))
  cmp$set_state(s[, , 3])
  expect_equal(cmp$compare_data(), x[, 3])
})


test_that("resample multi", {
  res <- dust_example("variable")
  obj <- res$new(list(list(len = 5), list(len = 5)), 0, 7,
                 seed = 1L, pars_multi = TRUE)
  m <- obj$state()
  m[] <- seq_along(m)
  obj$set_state(m)

  rng <- dust_rng$new(obj$rng_state(), 14)
  expect_identical(rng$state(), obj$rng_state())

  w <- cbind(runif(7), runif(7))
  idx <- obj$resample(w)
  expect_true(all(idx >= 1 & idx <= 7))
  expect_true(all(diff(idx) >= 0))

  ## Resampled state reflect index:
  s <- obj$state()
  expect_equal(s[, , 1], m[, idx[, 1], 1])
  expect_equal(s[, , 2], m[, idx[, 2], 2])

  ## Index is expected:
  u <- rng$unif_rand(14)[c(1, 8)]
  expect_equal(
    idx,
    cbind(resample_index(w[, 1], u[1]), resample_index(w[, 2], u[2])))
})


test_that("resample multi validates inputs", {
  res <- dust_example("variable")
  obj <- res$new(list(list(len = 5), list(len = 5)), 0, 7,
                 seed = 1L, pars_multi = TRUE)
  m <- obj$state()
  m[] <- seq_along(m)
  obj$set_state(m)

  rng <- dust_rng$new(obj$rng_state(), 14)
  expect_identical(rng$state(), obj$rng_state())

  w <- cbind(runif(7), runif(7))
  expect_error(
    obj$resample(c(w)),
    "Expected a matrix for 'weights', but given vector")
  expect_error(
    obj$resample(w[-1, ]),
    "Expected a matrix with 7 rows for 'weights'")
  expect_error(
    obj$resample(w[, -1, drop = FALSE]),
    "Expected a matrix with 2 columns for 'weights'")
  expect_error(
    obj$resample(array(w, c(dim(w), 1))),
    "Expected a matrix for 'weights'")

  ## Unchanged:
  expect_identical(obj$state(), m)
})
