test_that("can integrate logistic", {
  y0 <- c(1, 1)
  r <- c(0.1, 0.2)
  k <- c(100, 100)
  times <- 0:25

  analytic <- logistic_analytic(r, k, times, y0)
  dde <- logistic_dde(r, k, times, y0)

  ex <- dust_example("logistic")
  gen <- ex$generator
  pars <- list(r = r, K = k)
  n_particles <- 5
  mod <- gen$new(pars, 0, n_particles)

  actual <- vapply(times, function(t) mod$run(t),
                   matrix(0.0, 3, n_particles))
  expect_equal(actual[1:2, 1, ], analytic, tolerance = 1e-7)
  expect_equal(actual[1:2, 1, ], dde, tolerance = 1e-7)
  expect_identical(actual, actual[, rep(1, 5), ])
})


test_that("Can cope with systems that do not set all derivatives", {
  path <- dust_file("examples/ode/stochastic.cpp")
  code <- readLines(path)
  i <- grep("dydt\\[.+\\] = 0;", code)
  stopifnot(length(i) == 1)
  tmp <- tempfile(fileext = ".cpp")
  writeLines(code[-i], tmp)
  gen <- dust(tmp, quiet = TRUE)

  pars <- list(r1 = 0.1, r2 = 0.2, K1 = 100, K2 = 100, v = 1)
  mod1 <- gen$new(pars, 0, 1)
  y1 <- vector("list", 3)
  for (i in seq_along(y1)) {
    y1[[i]] <- mod1$run(i)
  }

  mod2 <- gen$new(pars, 0, 1)
  y2 <- vector("list", 3)
  for (i in seq_along(y1)) {
    mod2$reorder(1)
    y2[[i]] <- mod2$run(i)
  }

  expect_equal(y2, y1)
})


## A better test here would be possible if we had a model that had a
## step size that varied a bunch...
test_that("Error if step size becomes too small", {
  ex <- example_logistic()
  gen <- ex$generator
  pars <- ex$pars
  control <- dust_ode_control(step_size_min = 0.1)
  n_particles <- 5
  mod <- gen$new(pars, 0, n_particles, ode_control = control)
  expect_error(
    mod$run(5),
    "step too small")
  expect_error(
    mod$run(5),
    "Errors pending; reset required")
  mod$update_state(pars = ex$pars, reset_step_size = TRUE)
  expect_error(
    mod$run(5),
    "step too small")
})
