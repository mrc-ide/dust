test_that("create trivial multi dust object", {
  res <- dust_example("walk")
  obj1 <- res$new(list(sd = 1), 0, 10, seed = 1L, pars_multi = FALSE)
  obj2 <- res$new(list(list(sd = 1)), 0, 10, seed = 1L, pars_multi = TRUE)

  expect_identical(obj2$name(), obj1$name())
  expect_identical(obj2$param(), obj1$param())
  expect_identical(obj2$n_threads(), obj1$n_threads())
  expect_identical(obj2$has_openmp(), obj1$has_openmp())
  expect_identical(obj2$time(), obj1$time())

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
  expect_equal(obj$time(), ns)

  obj$update_state(pars = pars2, time = 0)
  expect_equal(obj$time(), 0)
  y2 <- obj$run(ns)
  expect_equal(obj$time(), ns)

  ## Then draw the random numbers:
  rng <- dust_rng$new(1, nd * np)
  m1 <- array(rng$normal(ns, 0, 1), c(ns, np, nd))
  m2 <- array(rng$normal(ns, 0, 1), c(ns, np, nd))

  expect_equal(
    matrix(y1, np, nd),
    apply(m1, 2:3, sum) * rep(c(sd1, sd2), each = np))

  expect_equal(
    matrix(y2, np, nd),
    apply(m2, 2:3, sum) * rep(c(sd2, sd3), each = np))
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

  mod$update_state(state = y)
  expect_equal(mod$state(), y)

  expect_error(
    mod$update_state(state = c(y)),
    "Expected array of rank 2 or 3 for 'state'")
  expect_error(
    mod$update_state(state = matrix(c(y), c(ns, nd * np))),
    "Expected a matrix with 3 cols for 'state' but given 39")
  expect_error(
    mod$update_state(state = y[-1, , ]),
    "Expected dimension 1 of 'state' to be 7 but given 6")
  expect_error(
    mod$update_state(state = y[, -1, ]),
    "Expected dimension 2 of 'state' to be 13 but given 12")
  expect_error(
    mod$update_state(state = y[, , -1]),
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
  mod$update_state(state = y)
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
  mod$update_state(state = y)

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
  mod$update_state(state = y)

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
  mod$update_state(state = y)

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
  filter_seed <- obj$rng_state(last_only = TRUE)
  y1 <- obj$run(1)

  a <- res$new(p1[[1]], 0, 10L, seed = c(seed[1:320], filter_seed))
  b <- res$new(p1[[2]], 0, 10L, seed = c(seed[321:640], filter_seed))
  expect_equal(drop(a$run(1)), y1[, , 1])
  expect_equal(drop(b$run(1)), y1[, , 2])

  expect_identical(obj$rng_state()[1:640],
                   c(a$rng_state()[1:320], b$rng_state()[1:320]))
  expect_identical(obj$rng_state(last_only = TRUE),
                   a$rng_state(last_only = TRUE))
  expect_identical(obj$rng_state(last_only = TRUE),
                   b$rng_state(last_only = TRUE))

  obj$update_state(pars = p2, set_initial_state = FALSE)
  expect_equal(obj$state(), y1)
  expect_equal(obj$time(), 1)
  expect_equal(obj$pars(), p2)

  y2 <- obj$run(2)
  a$update_state(pars = p2[[1]], set_initial_state = FALSE)
  b$update_state(pars = p2[[2]], set_initial_state = FALSE)
  expect_equal(drop(a$run(2)), y2[, , 1])
  expect_equal(drop(b$run(2)), y2[, , 2])

  expect_identical(obj$rng_state()[1:640],
                   c(a$rng_state()[1:320], b$rng_state()[1:320]))
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


## TODO: These no longer report the group they're in
test_that("Can't change parameter size on reset or set_pars", {
  res <- dust_example("variable")
  pars <- rep(list(list(len = 7)), 5)
  obj <- res$new(pars, 0, 10, seed = 1L, pars_multi = TRUE)
  pars2 <- rep(list(list(len = 8)), 5)
  expect_error(
    obj$update_state(pars = pars2, time = 0),
    paste("'pars' created inconsistent state size:",
          "expected length 7 but created length 8"))
  expect_error(
    obj$update_state(pars2, set_initial_state = FALSE),
    paste("'pars' created inconsistent state size:",
          "expected length 7 but created length 8"))
  pars3 <- pars
  pars3[[3]] <- pars2[[3]]
  expect_error(
    obj$update_state(pars = pars3, time = 0),
    paste("'pars' created inconsistent state size:",
          "expected length 7 but created length 8"))
  expect_error(
    obj$update_state(pars3, set_initial_state = FALSE),
    paste("'pars' created inconsistent state size:",
          "expected length 7 but created length 8"))
})


test_that("Validate parameter suitability", {
  res <- dust_example("variable")
  pars <- rep(list(list(len = 7)), 6)
  obj <- res$new(pars, 0, 10, seed = 1L, pars_multi = TRUE)

  expect_error(obj$update_state(pars = pars[-1], time = 0),
               "Expected a list of length 6 for 'pars'")
  expect_error(obj$update_state(pars[-1], set_initial_state = FALSE),
               "Expected a list of length 6 for 'pars'")

  expect_error(
    obj$update_state(pars[[1]], set_initial_state = FALSE),
    "Expected an unnamed list for 'pars' (given 'pars_multi')",
    fixed = TRUE)
  expect_error(
    obj$update_state(pars = pars[[1]], time = 0),
    "Expected an unnamed list for 'pars' (given 'pars_multi')",
    fixed = TRUE)

  pars2 <- structure(pars, dim = c(2, 3))
  expect_error(
    obj$update_state(pars2, set_initial_state = FALSE),
    "Expected a list with no dimension attribute for 'pars'")
  expect_error(
    obj$update_state(pars = pars2, time = 0),
    "Expected a list with no dimension attribute for 'pars'")
})


test_that("compare with multi pars", {
  res <- dust_example("sir")

  np <- 10
  end <- 150 * 4
  times <- seq(0, end, by = 4)
  ans <- res$new(list(), 0, np, seed = 1L)$simulate(times)
  d <- data.frame(time = times, incidence = ans[5, 1, ])

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
  cmp$update_state(state = s[, , 1])
  expect_equal(cmp$compare_data(), x[, 1])
  cmp$update_state(state = s[, , 2])
  expect_equal(cmp$compare_data(), x[, 2])
  cmp$update_state(state = s[, , 3])
  expect_equal(cmp$compare_data(), x[, 3])
})


test_that("validate setting data by length", {
  res <- dust_example("sir")

  np <- 10
  end <- 150 * 4
  times <- seq(0, end, by = 4)
  ans <- res$new(list(), 0, np, seed = 1L)$simulate(times)
  d <- data.frame(time = times, incidence = ans[5, 1, ])

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
  times <- seq(0, end, by = 4)
  ans <- res$new(list(), 0, np, seed = 1L)$simulate(times)
  d <- data.frame(time = times,
                  group = factor(rep(c("a", "b", "c"), each = length(times))),
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
  cmp$update_state(state = s[, , 1])
  expect_equal(cmp$compare_data(), x[, 1])

  cmp$set_data(dust_data(d[d$group == "b", ]))
  cmp$update_state(state = s[, , 2])
  expect_equal(cmp$compare_data(), x[, 2])

  cmp$set_data(dust_data(d[d$group == "c", ]))
  cmp$update_state(state = s[, , 3])
  expect_equal(cmp$compare_data(), x[, 3])
})


test_that("resample multi", {
  res <- dust_example("variable")
  obj <- res$new(list(list(len = 5), list(len = 5)), 0, 7,
                 seed = 1L, pars_multi = TRUE)
  m <- obj$state()
  m[] <- seq_along(m)
  obj$update_state(state = m)

  rng <- dust_rng$new(obj$rng_state(), 15)
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
  u <- rng$random_real(30)[, 15]
  expect_equal(
    idx,
    cbind(resample_index(w[, 1], u[1]), resample_index(w[, 2], u[2])))
})


test_that("resample multi with zero weights everywhere does nothing", {
  res <- dust_example("variable")
  np <- 31
  obj <- res$new(list(list(len = 5), list(len = 5)), 0, np,
                 seed = 1L, pars_multi = TRUE)
  m <- obj$state()
  m[] <- seq_along(m)
  obj$update_state(state = m)
  rng <- dust_rng$new(obj$rng_state(last_only = TRUE))
  idx <- obj$resample(matrix(0, np, 2))
  expect_equal(idx, cbind(seq_len(np), seq_len(np)))
  expect_equal(obj$state(), m)
  ## RNG state is the same after drawing two samples:
  rng$random_real(2)
  expect_identical(obj$rng_state(last_only = TRUE), rng$state())
})


test_that("resample multi validates inputs", {
  res <- dust_example("variable")
  obj <- res$new(list(list(len = 5), list(len = 5)), 0, 7,
                 seed = 1L, pars_multi = TRUE)
  m <- obj$state()
  m[] <- seq_along(m)
  obj$update_state(state = m)

  rng <- dust_rng$new(obj$rng_state(), 15)
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
  expect_silent(mod$update_state(state = s))
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
  expect_silent(mod$update_state(state = s))

  s2 <- s
  s2[] <- runif(length(s2))
  expect_error(mod$update_state(state = c(s2)),
               "Expected array of rank 3 or 4 for 'state'")
})


test_that("Can set pars into unreplicated multiparameter model", {
  p1 <- lapply(runif(10), function(x) list(len = 7, sd = x))
  p2 <- lapply(runif(10), function(x) list(len = 7, sd = x))
  res <- dust_example("variable")
  mod <- res$new(p1, 0, NULL, seed = 1L, pars_multi = TRUE)
  expect_silent(mod$update_state(pars = p2, set_initial_state = FALSE))
  expect_identical(mod$pars(), p2)

  mod$update_state(state = matrix(0, 7, 10))

  y <- mod$run(1)

  cmp <- dust_rng$new(1L, 10)$normal(7, 0, 1)
  expect_equal(y,
               matrix(t(cmp) * vapply(p2, "[[", 1, "sd"), 7, 10, TRUE))

  reshape <- function(p) {
    structure(p, dim = c(5, 2))
  }
  mod2 <- res$new(reshape(p1), 0, NULL, seed = 1L, pars_multi = TRUE)
  mod2$pars()
  mod2$update_state(state = array(0, c(7, 5, 2)))

  expect_silent(
    mod2$update_state(pars = reshape(p2), set_initial_state = FALSE))
  expect_identical(mod2$pars(), reshape(p2))
  y2 <- mod2$run(1)
  expect_equal(dim(y2), c(7, 5, 2))
  expect_equal(c(y2), c(y))
})


test_that("Can set pars into a structured parameter model", {
  p1 <- lapply(runif(10), function(x) list(len = 7, sd = x))
  p2 <- lapply(runif(10), function(x) list(len = 7, sd = x))
  p3 <- lapply(runif(10), function(x) list(len = 7, sd = x))
  dim(p1) <- dim(p2) <- c(5, 2)

  res <- dust_example("variable")
  mod <- res$new(p1, 0, 6, seed = 1L, pars_multi = TRUE)

  expect_silent(mod$update_state(p2, set_initial_state = FALSE))
  expect_identical(mod$pars(), p2)

  expect_error(mod$update_state(p3, set_initial_state = FALSE),
               "Expected a list array of rank 2 for 'pars'")
  expect_error(mod$update_state(p2[-1, ], set_initial_state = FALSE),
               "Expected dimension 1 of 'pars' to be 5 but given 4")
})


test_that("Can set state into a multiparameter model, shared + flat", {
  res <- dust_example("variable")
  p <- lapply(runif(6), function(x) list(len = 5, sd = x))
  mod <- res$new(p, 0, 7, pars_multi = TRUE)

  s <- mod$state()
  y <- array(as.numeric(seq_along(s)), dim(s))

  ## Set full model state:
  expect_silent(mod$update_state(state = y))
  expect_identical(mod$state(), y)

  ## Replicate model state:
  mod$update_state(state = y[, 1, ])
  expect_identical(mod$state(), y[, rep(1, 7), ])
})


test_that("Can set state into a multiparameter model, unshared + flat", {
  res <- dust_example("variable")
  p <- lapply(runif(6), function(x) list(len = 5, sd = x))
  mod <- res$new(p, 0, NULL, pars_multi = TRUE)

  s <- mod$state()
  y <- array(as.numeric(seq_along(s)), dim(s))

  ## Set full model state:
  expect_silent(mod$update_state(state = y))
  expect_identical(mod$state(), y)

  expect_error(mod$update_state(state = y[, 1]),
               "Expected a matrix for 'state'")
  expect_error(mod$update_state(state = y[, 1, drop = FALSE]),
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
  expect_silent(mod$update_state(state = y))
  expect_identical(mod$state(), y)

  ## Replicate model state:
  mod$update_state(state = y[, 1, , ])
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
  expect_silent(mod$update_state(state = y))
  expect_identical(mod$state(), y)

  ## Replicate model state:
  expect_error(mod$update_state(state = y[, 1, ]),
               "Expected an array of rank 3 for 'state'")
  expect_error(mod$update_state(state = y[, 1, , drop = FALSE]),
               "Expected dimension 2 of 'state' to be 3 but given 1")
})


test_that("Reorder across particles", {
  res <- dust_example("variable")
  p <- lapply(runif(6), function(x) list(len = 5, sd = x))
  mod <- res$new(p, 0, NULL, pars_multi = TRUE)

  s <- mod$state()
  y <- array(as.numeric(seq_along(s)), dim(s))
  mod$update_state(state = y)

  i <- sample.int(length(p), replace = TRUE)
  mod$reorder(i)
  expect_equal(mod$state(), y[, i])
})


test_that("share data across parameter sets", {
  res <- dust_example("sir")

  np <- 10
  end <- 150 * 4
  times <- seq(0, end, by = 4)
  ans <- res$new(list(), 0, np, seed = 1L)$simulate(times)
  d <- data.frame(time = times, incidence = ans[5, 1, ])

  ## Use Inf for exp_noise as that gives us deterministic results
  p <- list(exp_noise = Inf)
  mod <- res$new(rep(list(p), 3), 0, np, seed = 1L, pars_multi = TRUE)
  s <- mod$run(36)
  expect_null(mod$compare_data())

  expect_error(mod$set_data(dust_data(d)), "Expected a list of length 4")
  mod$set_data(dust_data(d), shared = TRUE)

  ll <- mod$compare_data()
  expect_equal(dim(ll), c(10, 3))

  ## Confirm that we match the multi-data case:
  mod$set_data(dust_data(d, multi = 3))
  expect_identical(ll, mod$compare_data())
})


test_that("Can trivial create multi-parameter-set ode object", {
  ex <- example_logistic()
  pars <- ex$pars
  gen <- ex$generator
  np <- 10
  obj1 <- gen$new(pars, 0, np, seed = 1L, pars_multi = FALSE)
  obj2 <- gen$new(list(pars), 0, np, seed = 1L, pars_multi = TRUE)

  expect_identical(obj2$name(), obj1$name())
  expect_identical(obj2$param(), obj1$param())
  expect_identical(obj2$n_threads(), obj1$n_threads())
  expect_identical(obj2$has_openmp(), obj1$has_openmp())
  expect_identical(obj2$time(), obj1$time())

  expect_equal(obj2$pars(), list(obj1$pars()))
  expect_equal(obj2$info(), list(obj1$info()))
  expect_equal(obj2$shape(), c(obj1$shape(), 1))

  expect_identical(obj2$rng_state(), obj1$rng_state())

  expect_identical(obj1$n_pars(), 0L)
  expect_identical(obj2$n_pars(), 1L)

  expect_identical(obj2$state(), array(obj1$state(), c(3, np, 1)))
  expect_identical(obj2$state(1L), array(obj1$state(1L), c(1, np, 1)))

  y1 <- obj1$run(1)
  y2 <- obj2$run(1)
  expect_equal(c(y1), c(y2))
  expect_equal(dim(y2), c(3, np, 1))
})


test_that("Can set parameters into a trivial multi-parmeter ode object", {
  gen <- dust_example("logistic")

  pars1 <- list(r = c(0.1, 0.2), K = c(100, 200))
  pars2 <- list(r = c(0.3, 0.4), K = c(100, 200))

  np <- 10
  obj1 <- gen$new(pars1, 0, np, seed = 1L, pars_multi = FALSE)
  obj2 <- gen$new(list(pars1), 0, np, seed = 1L, pars_multi = TRUE)

  y1 <- obj1$run(1)
  y2 <- obj2$run(1)
  expect_equal(y2, array(y1, c(dim(y1), 1)))

  obj1$update_state(pars = pars2)
  obj2$update_state(pars = list(pars2))

  y1 <- obj1$run(10)
  y2 <- obj2$run(10)
  expect_equal(y2, array(y1, c(dim(y1), 1)))
})
