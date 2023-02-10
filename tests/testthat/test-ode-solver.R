test_that("can integrate logistic", {
  y0 <- c(1, 1)
  r <- c(0.1, 0.2)
  k <- c(100, 100)
  times <- 0:25

  analytic <- logistic_analytic(r, k, times, y0)
  dde <- logistic_dde(r, k, times, y0)

  path <- dust_file("examples/ode/logistic.cpp")
  gen <- dust(path, quiet = TRUE)
  pars <- list(r1 = r[[1]], r2 = r[[2]], K1 = k[[1]], K2 = k[[2]])
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
