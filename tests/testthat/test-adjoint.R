test_that("can run adjoint model", {
  skip_for_compilation()
  gen <- dust(dust_file("examples/adjoint/sir.cpp"), quiet = TRUE)
  incidence <- data_frame(
    time = (1:10) * 4,
    incidence = c(3, 2, 2, 2, 1, 3, 2, 5, 5, 6))
  d <- dust::dust_data(incidence)
  pars <- list(beta = 0.25, gamma = 0.1, I0 = 1)
  mod <- gen$new(pars, 0, 1, deterministic = TRUE)
  mod$set_data(d)
  res <- mod$run_adjoint()
  expect_equal(res$log_likelihood, -44.0256051296862, tolerance = 1e-14)
  expect_equal(res$gradient,
               c(beta = 244.877646917118,
                 gamma = -140.566517375877,
                 I0 = 25.2152128116894),
               tolerance = 1e-14)
})


test_that("check that adjoint model can't be run for impossible cases", {
  gen <- dust(dust_file("examples/adjoint/sir.cpp"), quiet = TRUE)
  incidence <- data_frame(
    time = (1:10) * 4,
    incidence = c(3, 2, 2, 2, 1, 3, 2, 5, 5, 6))
  d <- dust::dust_data(incidence)
  pars <- list(beta = 0.25, gamma = 0.1, I0 = 1)

  expect_error(
    gen$new(pars, 0, 10, deterministic = FALSE)$run_adjoint(),
    "'run_adjoint()' only allowed for deterministic models",
    fixed = TRUE)
  expect_error(
    gen$new(pars, 0, 10, deterministic = TRUE)$run_adjoint(),
    "'run_adjoint()' only allowed with single particle",
    fixed = TRUE)
  expect_error(
    gen$new(pars, 0, 1, deterministic = TRUE)$run_adjoint(),
    "Data has not been set for this object",
    fixed = TRUE)

  mod <- gen$new(pars, 0, 1, deterministic = TRUE)
  mod$set_data(d)
  mod$update_state(time = 8)
  expect_error(
    mod$run_adjoint(),
    "Expected model start time (8) to be at most the first data time (4)",
    fixed = TRUE)
})
