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
  expect_equal(c(y1), c(y2))
  expect_equal(dim(y2), c(1, 10, 1))
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

  expect_equal(obj1$n_particles(), 10)
  expect_equal(obj2$n_particles(), 10)
  expect_equal(obj1$n_particles_each(), 10)
  expect_equal(obj2$n_particles_each(), 5)
})


test_that("Can reset particles and resume continues with rng", {
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
    "Expected array of rank 2 or 3 for 'state'")
  expect_error(
    mod$set_state(matrix(c(y), c(ns, nd * np))),
    "Expected a matrix with 3 cols for 'state' but given 39")
  expect_error(
    mod$set_state(y[-1, , ]),
    "Expected dimension 1 of 'state' to be 7 but given 6")
  expect_error(
    mod$set_state(y[, -1, ]),
    "Expected dimension 2 of 'state' to be 13 but given 12")
  expect_error(
    mod$set_state(y[, , -1]),
    "Expected dimension 3 of 'state' to be 3 but given 2")

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
    "Expected a matrix with 13 rows for 'index' but given 12")
  expect_error(
    mod$reorder(i[, -1]),
    "Expected a matrix with 3 cols for 'index' but given 2")
  i[1] <- np + 1
  expect_error(
    mod$reorder(i),
    "All elements of 'index' must lie in [1, 13]",
    fixed = TRUE)
  expect_equal(mod$state(), y)
})


test_that("Can change pars", {
  res <- dust_example("walk")

  p1 <- list(list(sd = 1), list(sd = 10))
  p2 <- list(list(sd = 2), list(sd = 0)) # this is really easy see the failure

  obj <- res$new(p1, 0, 10L, pars_multi = TRUE)
  seed <- obj$rng_state()
  y1 <- obj$run(1)

  a <- res$new(p1[[1]], 0, 10L, seed = seed[1:320])
  b <- res$new(p1[[2]], 0, 10L, seed = seed[321:640])
  expect_equal(drop(a$run(1)), y1[, , 1])
  expect_equal(drop(b$run(1)), y1[, , 2])

  expect_identical(obj$rng_state(), c(a$rng_state(), b$rng_state()))

  obj$set_pars(p2)
  expect_equal(obj$state(), y1)
  expect_equal(obj$step(), 1)
  expect_equal(obj$pars(), p2)

  y2 <- obj$run(2)
  a$set_pars(p2[[1]])
  b$set_pars(p2[[2]])
  expect_equal(drop(a$run(2)), y2[, , 1])
  expect_equal(drop(b$run(2)), y2[, , 2])

  expect_identical(obj$rng_state(), c(a$rng_state(), b$rng_state()))
})


test_that("must use same sized simulations", {
  res <- dust_example("variable")
  pars <- list(list(len = 7), list(len = 8))
  expect_error(
    res$new(pars, 0, 10, seed = 1L, pars_multi = TRUE),
    paste("'pars' created inconsistent state size:",
          "expected length 7 but parameter set 2 created length 8"),
    fixed = TRUE)
})


test_that("Can't change parameter size on reset or set_pars", {
  res <- dust_example("variable")
  pars <- rep(list(list(len = 7)), 5)
  obj <- res$new(pars, 0, 10, seed = 1L, pars_multi = TRUE)
  pars2 <- rep(list(list(len = 8)), 5)
  expect_error(
    obj$reset(pars2, 0),
    paste("'pars' created inconsistent state size:",
          "expected length 7 but parameter set 1 created length 8"))
  expect_error(
    obj$set_pars(pars2),
    paste("'pars' created inconsistent state size:",
          "expected length 7 but parameter set 1 created length 8"))
  pars3 <- pars
  pars3[[3]] <- pars2[[3]]
  expect_error(
    obj$reset(pars3, 0),
    paste("'pars' created inconsistent state size:",
          "expected length 7 but parameter set 3 created length 8"))
  expect_error(
    obj$set_pars(pars3),
    paste("'pars' created inconsistent state size:",
          "expected length 7 but parameter set 3 created length 8"))
})


test_that("Validate parameter suitability", {
  res <- dust_example("variable")
  pars <- rep(list(list(len = 7)), 6)
  obj <- res$new(pars, 0, 10, seed = 1L, pars_multi = TRUE)

  expect_error(obj$reset(pars[-1], 0),
               "Expected a list of length 6 for 'pars'")
  expect_error(obj$set_pars(pars[-1]),
               "Expected a list of length 6 for 'pars'")

  expect_error(
    obj$set_pars(pars[[1]]),
    "Expected an unnamed list for 'pars' (given 'pars_multi')",
    fixed = TRUE)
  expect_error(
    obj$reset(pars[[1]], 0),
    "Expected an unnamed list for 'pars' (given 'pars_multi')",
    fixed = TRUE)

  pars2 <- structure(pars, dim = c(2, 3))
  expect_error(
    obj$set_pars(pars2),
    "Expected a list with no dimension attribute for 'pars'")
  expect_error(
    obj$reset(pars2, 0),
    "Expected a list with no dimension attribute for 'pars'")
})


test_that("compare with multi pars", {
  res <- dust_example("sir")

  np <- 10
  end <- 150 * 4
  steps <- seq(0, end, by = 4)
  ans <- res$new(list(), 0, np, seed = 1L)$simulate(steps)
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


test_that("validate setting data by length", {
  res <- dust_example("sir")

  np <- 10
  end <- 150 * 4
  steps <- seq(0, end, by = 4)
  ans <- res$new(list(), 0, np, seed = 1L)$simulate(steps)
  d <- data.frame(step = steps, incidence = ans[5, 1, ])

  mod <- res$new(rep(list(list()), 3), 0, np, seed = 1L, pars_multi = TRUE)
  expect_error(
    mod$set_data(dust_data(d, multi = 4)),
    "Expected a list of length 4 for element 1 of 'data'")

  dat <- dust_data(d, multi = 3)
  dat[[50]] <- dat[[50]][-4]
  expect_error(
    mod$set_data(dat),
    "Expected a list of length 4 for element 50 of 'data'")

  dat[[25]] <- list()
  expect_error(
    mod$set_data(dat),
    "Expected a list of length 4 for element 25 of 'data'")
})


test_that("compare with multi pars and different data", {
  res <- dust_example("sir")

  np <- 10
  end <- 150 * 4
  steps <- seq(0, end, by = 4)
  ans <- res$new(list(), 0, np, seed = 1L)$simulate(steps)
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
    "Expected a matrix for 'weights'")
  expect_error(
    obj$resample(w[-1, ]),
    "Expected a matrix with 7 rows for 'weights' but given 6")
  expect_error(
    obj$resample(w[, -1, drop = FALSE]),
    "Expected a matrix with 2 cols for 'weights' but given 1")
  expect_error(
    obj$resample(array(w, c(dim(w), 1))),
    "Expected a matrix for 'weights'")

  ## Unchanged:
  expect_identical(obj$state(), m)
})


test_that("must create at least one element", {
  res <- dust_example("variable")
  expect_error(res$new(list(), 0, 7, seed = 1L, pars_multi = TRUE),
               "Expected 'pars' to have at least one element")
})


test_that("must use an unnamed list", {
  res <- dust_example("variable")
  pars <- list(a = list(len = 5), b = list(len = 5))
  expect_error(
    res$new(pars, 0, 7, pars_multi = TRUE),
    "Expected an unnamed list for 'pars' (given 'pars_multi')",
    fixed = TRUE)
})


test_that("Can create unreplicated multi-pars examples", {
  p <- lapply(runif(10), function(x) list(len = 7, sd = x))
  res <- dust_example("variable")
  mod <- res$new(p, 0, NULL, seed = 1L, pars_multi = TRUE)
  s <- mod$state()
  expect_equal(s, matrix(1:7, 7, 10))

  s[] <- runif(length(s))
  expect_silent(mod$set_state(s))
  expect_identical(mod$state(), s)

  expect_equal(mod$shape(), 10)

  s <- mod$simulate(0:5)
  expect_equal(dim(s), c(7, 10, 6))
  expect_equal(mod$n_particles_each(), 1)
  expect_equal(mod$n_particles(), 10)
})


test_that("Can set state into 3d model", {
  p <- lapply(runif(10), function(x) list(len = 7, sd = x))
  dim(p) <- c(2, 5)
  res <- dust_example("variable")
  mod <- res$new(p, 0, 3, seed = 1L, pars_multi = TRUE)
  s <- mod$state()
  expect_equal(dim(s), c(7, 3, 2, 5)) # n_state, n_particles, dim(pars)
  expect_equal(mod$shape(), c(3, 2, 5))
  expect_equal(mod$n_particles_each(), 3)
  expect_equal(mod$n_particles(), 3 * 2 * 5)

  s <- mod$state()
  s[] <- runif(length(s))
  expect_silent(mod$set_state(s))

  s2 <- s
  s2[] <- runif(length(s2))
  expect_error(mod$set_state(c(s2)),
               "Expected array of rank 3 or 4 for 'state'")
})


test_that("Can set pars into unreplicated multiparameter model", {
  p1 <- lapply(runif(10), function(x) list(len = 7, sd = x))
  p2 <- lapply(runif(10), function(x) list(len = 7, sd = x))
  res <- dust_example("variable")
  mod <- res$new(p1, 0, NULL, seed = 1L, pars_multi = TRUE)
  expect_silent(mod$set_pars(p2))
  expect_identical(mod$pars(), p2)

  mod$set_state(matrix(0, 7, 10))

  y <- mod$run(1)

  cmp <- dust_rng$new(1L, 10)$norm_rand(7 * 10)
  expect_equal(y,
               matrix(cmp * vapply(p2, "[[", 1, "sd"), 7, 10, TRUE))

  reshape <- function(p) {
    structure(p, dim = c(5, 2))
  }
  mod2 <- res$new(reshape(p1), 0, NULL, seed = 1L, pars_multi = TRUE)
  mod2$pars()
  mod2$set_state(array(0, c(7, 5, 2)))

  expect_silent(mod2$set_pars(reshape(p2)))
  expect_identical(mod2$pars(), reshape(p2))
  y2 <- mod2$run(1)
  expect_equal(dim(y2), c(7, 5, 2))
  expect_equal(c(y2), c(y))
})


test_that("Can set state into a multiparameter model, shared + flat", {
  res <- dust_example("variable")
  p <- lapply(runif(6), function(x) list(len = 5, sd = x))
  mod <- res$new(p, 0, 7, pars_multi = TRUE)

  s <- mod$state()
  y <- array(as.numeric(seq_along(s)), dim(s))

  ## Set full model state:
  expect_silent(mod$set_state(y))
  expect_identical(mod$state(), y)

  ## Replicate model state:
  mod$set_state(y[, 1, ])
  expect_identical(mod$state(), y[, rep(1, 7), ])
})


test_that("Can set state into a multiparameter model, unshared + flat", {
  res <- dust_example("variable")
  p <- lapply(runif(6), function(x) list(len = 5, sd = x))
  mod <- res$new(p, 0, NULL, pars_multi = TRUE)

  s <- mod$state()
  y <- array(as.numeric(seq_along(s)), dim(s))

  ## Set full model state:
  expect_silent(mod$set_state(y))
  expect_identical(mod$state(), y)

  expect_error(mod$set_state(y[, 1]),
               "Expected a matrix for 'state'")
  expect_error(mod$set_state(y[, 1, drop = FALSE]),
               "Expected a matrix with 6 cols for 'state' but given 1")
})


test_that("Can set state into a multiparameter model, shared + structured", {
  res <- dust_example("variable")
  p <- lapply(runif(6), function(x) list(len = 5, sd = x))
  dim(p) <- 3:2
  mod <- res$new(p, 0, 7, pars_multi = TRUE)

  s <- mod$state()
  y <- array(as.numeric(seq_along(s)), dim(s))

  ## Set full model state:
  expect_silent(mod$set_state(y))
  expect_identical(mod$state(), y)

  ## Replicate model state:
  mod$set_state(y[, 1, , ])
  expect_identical(mod$state(), y[, rep(1, 7), , ])
})


test_that("Can set state into a multiparameter model, unshared + structured", {
  res <- dust_example("variable")
  p <- lapply(runif(6), function(x) list(len = 5, sd = x))
  dim(p) <- 3:2
  mod <- res$new(p, 0, NULL, pars_multi = TRUE)

  s <- mod$state()
  y <- array(as.numeric(seq_along(s)), dim(s))

  ## Set full model state:
  expect_silent(mod$set_state(y))
  expect_identical(mod$state(), y)

  ## Replicate model state:
  expect_error(mod$set_state(y[, 1, ]),
               "Expected an array of rank 3 for 'state'")
  expect_error(mod$set_state(y[, 1, , drop = FALSE]),
               "Expected dimension 2 of 'state' to be 3 but given 1")
})
