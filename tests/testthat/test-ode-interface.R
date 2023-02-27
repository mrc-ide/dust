test_that("Can compile a simple model", {
  ex <- example_logistic()
  n_particles <- 10
  mod <- ex$generator$new(ex$pars, pi, n_particles)
  expect_s3_class(mod, "dust")
  expect_equal(mod$time(), pi)
  expect_equal(mod$pars(), ex$pars)
  expect_equal(mod$info(), list(n = 2))
  expect_equal(mod$n_particles(), n_particles)
  expect_equal(mod$shape(), n_particles)
  expect_equal(mod$n_particles_each(), n_particles)
  expect_equal(mod$n_pars(), 0L)
  expected_control <- dust_ode_control(
    max_steps = 10000, rtol = 1e-6, atol = 1e-6,
    step_size_min = 1e-8, step_size_max = Inf,
    debug_record_step_times = FALSE)
  expect_equal(mod$ode_control(), expected_control)
})

test_that("Can compile a simple model with control", {
  ex <- example_logistic()
  n_particles <- 10
  control <- dust_ode_control(max_steps = 10, rtol = 0.01, atol = 0.02,
                          step_size_min = 0.1, step_size_max = 1,
                          debug_record_step_times = FALSE)
  mod <- ex$generator$new(ex$pars, pi, n_particles, ode_control = control)
  ctl <- mod$ode_control()
  expect_s3_class(control, "dust_ode_control")
  expect_s3_class(ctl, "dust_ode_control")
  expect_equal(ctl, control)
})

test_that("Can compile a simple model with partial control", {
  ex <- example_logistic()
  generate_control <- function(control) {
    ex$generator$new(ex$pars, pi, 10, ode_control = control)$ode_control()
  }

  control <- dust_ode_control(max_steps = 10, atol = 0.2)
  ctl <- generate_control(control)
  expect_s3_class(ctl, "dust_ode_control")
  expect_equal(ctl$max_steps, 10)
  expect_equal(ctl$atol, 0.2)
  expect_equal(ctl$rtol, 1e-6)
  expect_equal(ctl$step_size_min, 1e-8)
  expect_equal(ctl$step_size_max, Inf)
  expect_false(ctl$debug_record_step_times)

  default <- generate_control(NULL)
  expect_equal(ctl, modifyList(default, list(max_steps = 10, atol = 0.2)))

  ## A couple more
  expect_equal(generate_control(dust_ode_control(atol = 0.2)),
               modifyList(default, list(atol = 0.2)))
  expect_equal(
    generate_control(dust_ode_control(debug_record_step_times = FALSE)),
    modifyList(default, list(debug_record_step_times = FALSE)))
})

test_that("Returns full state from run when no index set", {
  ex <- example_logistic()
  n_particles <- 10
  mod <- ex$generator$new(ex$pars, 1, n_particles)
  res <- mod$run(2)
  expect_equal(dim(res), c(3, n_particles))

  state <- mod$state()
  expect_identical(res, state)
})

test_that("Returns state from run for set index", {
  ex <- example_logistic()
  n_particles <- 10
  mod <- ex$generator$new(ex$pars, 1, n_particles)
  mod$set_index(1)
  res <- mod$run(2)
  expect_equal(dim(res), c(1, n_particles))

  state <- mod$state()
  expect_identical(res, state[1, , drop = FALSE])
})

test_that("Can get arbitrary partial state", {
  ex <- example_logistic()
  n_particles <- 10
  mod <- ex$generator$new(ex$pars, 1, n_particles)
  res <- mod$run(2)
  expect_equal(dim(res), c(3, n_particles))

  state <- mod$state(1)
  expect_identical(state, res[1, , drop = FALSE])

  state <- mod$state(2)
  expect_identical(state, res[2, , drop = FALSE])

  state <- mod$state()
  expect_identical(state, res)
})

test_that("Error if initialised with no particles", {
  ex <- example_logistic()
  n_particles <- 10
  expect_error(ex$generator$new(ex$pars, 0, 0),
               "'n_particles' must be positive (was given 0)",
               fixed = TRUE)
  expect_error(ex$generator$new(ex$pars, 0, -1),
               "'n_particles' must be positive (was given -1)",
               fixed = TRUE)
})

test_that("Error if partial state index is invalid", {
  ex <- example_logistic()
  n_particles <- 10
  mod <- ex$generator$new(ex$pars, 1, n_particles)
  expect_error(mod$state(4),
               "All elements of 'index' must lie in [1, 3]",
               fixed = TRUE)
  expect_error(mod$state(c(1, 2, 4)),
               "All elements of 'index' must lie in [1, 3]",
               fixed = TRUE)
})

test_that("Can set vector index", {
  ex <- example_logistic()
  n_particles <- 10
  mod <- ex$generator$new(ex$pars, 1, n_particles)
  mod$set_index(c(1, 2))
  res <- mod$run(2)
  expect_equal(dim(res), c(2, n_particles))
})

test_that("Can retrieve index", {
  ex <- example_logistic()
  n_particles <- 10
  mod <- ex$generator$new(ex$pars, 1, n_particles)
  expect_equal(NULL, mod$index())
  mod$set_index(c(1, 2))
  expect_equal(c(1, 2), mod$index())
})

test_that("Setting a named index returns names", {
  ex <- example_logistic()
  r <- c(0.1, 0.2)
  k <- c(100, 100)
  pars <- list(r = r, K = k)
  n_particles <- 5
  mod <- ex$generator$new(pars, 0, n_particles)
  analytic <- logistic_analytic(r, k, 1, c(1, 1))
  mod$set_index(c(y1 = 1L, y2 = 2L))
  expect_equal(
    mod$run(1),
    rbind(y1 = rep(analytic[1, ], n_particles),
          y2 = rep(analytic[2, ], n_particles)),
    tolerance = 1e-7)
})

test_that("Can clear index", {
  ex <- example_logistic()
  r <- c(0.1, 0.2)
  k <- c(100, 100)
  pars <- list(r = r, K = k)
  n_particles <- 5
  mod <- ex$generator$new(pars, 0, n_particles)
  analytic <- logistic_analytic(r, k, 1:2, c(1, 1))
  mod$set_index(c(y1 = 1L))
  expect_equal(
    mod$run(1),
    rbind(y1 = rep(analytic[1, 1], n_particles)),
    tolerance = 1e-7)
  expect_equal(mod$index(), c(y1 = 1L))
  mod$set_index(NULL)
  expect_equal(
    mod$run(2)[1:2, ],
    rbind(rep(analytic[1, 2], n_particles),
          rep(analytic[2, 2], n_particles)),
    tolerance = 1e-7)
  expect_null(mod$index())
})

test_that("Cannot set invalid index", {
  ex <- example_logistic()
  n_particles <- 10
  mod <- ex$generator$new(ex$pars, 1, n_particles)
  expect_error(mod$set_index(0),
               "All elements of 'index' must lie in [1, 3]",
               fixed = TRUE)
  expect_error(mod$set_index(4),
               "All elements of 'index' must lie in [1, 3]",
               fixed = TRUE)
  expect_error(mod$set_index(c(1, 2, 4)),
               "All elements of 'index' must lie in [1, 3]",
               fixed = TRUE)
})

test_that("End time must be later than initial time", {
  ex <- example_logistic()
  n_particles <- 10
  initial_time <- 5
  mod <- ex$generator$new(ex$pars, initial_time, n_particles)
  expect_equal(mod$time(), initial_time)
  e <- "'time_end' must be at least 5"
  expect_error(mod$run(2), e, fixed = TRUE)
})

test_that("Can retrieve statistics", {
  ex <- example_logistic()
  n_particles <- 5
  mod <- ex$generator$new(ex$pars, 1, n_particles)
  stats <- mod$ode_statistics()
  expect_equal(dim(stats), c(3, n_particles))
  expect_equal(row.names(stats),
               c("n_steps", "n_steps_accepted", "n_steps_rejected"))
  expect_s3_class(stats, "ode_statistics")
  expect_true(all(stats == 0))
  lapply(1:10, function(t) mod$run(t))
  stats <- mod$ode_statistics()
  expect_true(all(stats == stats[, rep(1, n_particles)]))
  expect_true(all(stats["n_steps", ] > 0))

  expect_null(attr(stats, "step_times")) # see below
})

test_that("Can retrieve statistics", {
  ex <- example_logistic()
  n_particles <- 5
  mod <- ex$generator$new(ex$pars, 1, n_particles)
  stats <- mod$ode_statistics()
  expect_equal(dim(stats), c(3, n_particles))
  expect_equal(row.names(stats),
               c("n_steps", "n_steps_accepted", "n_steps_rejected"))
  expect_s3_class(stats, "ode_statistics")
  expect_true(all(stats == 0))
  lapply(1:10, function(t) mod$run(t))
  stats <- mod$ode_statistics()
  expect_true(all(stats == stats[, rep(1, n_particles)]))
  expect_true(all(stats["n_steps", ] > 0))
})

test_that("Can get model size", {
  ex <- example_logistic()
  mod <- ex$generator$new(ex$pars, 1, 1)
  expect_equal(mod$n_state(), 3)
  mod$set_index(1)
  expect_equal(mod$n_state(), 3)
  mod$set_index(c(1, 2, 3))
  expect_equal(mod$n_state(), 3)
})

test_that("can run to noninteger time", {
  ex <- example_logistic()
  mod <- ex$generator$new(ex$pars, 0, 1)

  t <- 3.95

  y <- mod$run(t)
  expect_equal(mod$time(), t)
  expect_equal(y[1:2, , drop = FALSE],
               logistic_analytic(c(0.1, 0.2), c(100, 100), t, c(1, 1)),
               tolerance = 1e-7)
})

test_that("Errors are reported", {
  ex <- example_logistic()
  mod <- ex$generator$new(ex$pars, 0, 2,
                          ode_control = dust_ode_control(max_steps = 1))
  err <- expect_error(mod$run(5), "2 particles reported errors.")
  expect_match(
    err$message,
    "- 1: too many steps")
})

test_that("Can run a stochastic model", {
  ex <- example_logistic()
  gen <- ex$generator

  np <- 10
  pars <- list(r = c(0, 0.2), K = c(100, 200), v = 0.1)
  mod <- gen$new(pars, 0, np, seed = 1L)
  mod$set_stochastic_schedule(0:5)

  rng <- dust::dust_rng$new(n_streams = np, seed = 1L)

  expect_equal(mod$state(), matrix(c(1, 1, 2), 3, np))

  ## Events happen at t + eps so running to t leaves things unchanged:
  mod$run(0)
  expect_equal(mod$state(), matrix(c(1, 1, 2), 3, np))

  ## Any bit beyond and we will run the stochastic update
  y1 <- mod$run(1e-8)
  r <- exp(rng$normal(2, 0, 0.1))
  expect_equal(y1, rbind(r, colSums(r)))

  ## Run up to the next one and we won't run a stochastic step, but we
  ## will complete a full deterministic step
  y2 <- mod$run(1)
  expect_equal(y2[1, ], y1[1, ]) # deterministic change turned off here
  expect_equal(y2[2, ],
               drop(logistic_analytic(0.2, 200, 1, y1[2, ])))
  expect_equal(y2[3, ], colSums(y2[1:2, ]))

  ## Up to the end we've run 5 stochastic updates (but not the 6th)
  y_end <- mod$run(5)
  rng <- dust::dust_rng$new(n_streams = np, seed = 1L)
  r <- array(rng$normal(2 * 5, 0, 0.1), c(2, 5, np))
  expect_equal(
    y_end[1, ],
    apply(exp(r[1, , ]), 2, prod))
  expect_equal(y_end[3, ], colSums(y_end[1:2, ]))
})


test_that("Can validate the stochastic schedule times", {
  ex <- example_logistic()
  gen <- ex$generator

  np <- 10
  pars <- list(r = c(0, 0.2), K = c(100, 200), v = 0.1)
  mod <- gen$new(pars, 0, np, seed = 1L)
  expect_error(
    mod$set_stochastic_schedule(c(1, 2, 3, 3, 4, 5)),
    "schedule must be strictly increasing; see time[4]",
    fixed = TRUE)
  expect_error(
    mod$set_stochastic_schedule(c(1, 2, 3, 4, 1, 2, 3)),
    "schedule must be strictly increasing; see time[5]",
    fixed = TRUE)

  ## No schedule set:
  y <- mod$run(10)
  expect_equal(y[1, ], rep(1, np))
})

test_that("A null schedule clears stochastic schedule", {
  ex <- example_logistic()
  gen <- ex$generator

  np <- 10
  pars <- list(r = c(0, 0.2), K = c(100, 200), v = 0.1)
  mod <- gen$new(pars, 0, np, seed = 1L)
  mod$set_stochastic_schedule(0:5)
  mod$set_index(1)

  rng <- dust::dust_rng$new(n_streams = np, seed = 1L)
  ## running draws 12 numbers per particle:
  y <- drop(mod$run(10))
  r <- array(rng$normal(2 * 6, 0, 0.1), c(2, 6, np))
  expect_equal(y, apply(exp(r[1, , ]), 2, prod))

  ## reset and rerun, draw another set:
  mod$update_state(time = 0, pars = pars, set_initial_state = TRUE)
  y <- drop(mod$run(10))
  r <- array(rng$normal(2 * 6, 0, 0.1), c(2, 6, np))
  expect_equal(y, apply(exp(r[1, , ]), 2, prod))

  mod$update_state(time = 0, pars = pars, set_initial_state = TRUE)
  mod$set_stochastic_schedule(NULL)
  y <- drop(mod$run(10))
  expect_equal(y, rep(1, np))
})

test_that("Basic threading test", {
  path <- dust_file("examples/ode/parallel.cpp")
  gen <- dust(path, quiet = TRUE)

  obj <- gen$new(list(sd = 1), 0, 10, n_threads = 2L, seed = 1L)
  obj$set_index(c(hasopenmp = 1L, threadnum = 2L))
  res <- obj$run(1)
  expect_true(all(res["hasopenmp", ] == obj$has_openmp()))
  if (obj$has_openmp()) {
    expect_equal(sum(res["threadnum", ] == 0), 5)
    expect_equal(sum(res["threadnum", ] == 1), 5)
  } else {
    expect_equal(sum(res["threadnum", ] == -1), 10)
  }
  ## And again without parallel
  obj <- gen$new(list(sd = 1), 0, 10, n_threads = 1L, seed = 1L)
  obj$set_index(c(hasopenmp = 1L, threadnum = 2L))
  res <- obj$run(1)
  expect_true(all(res["hasopenmp", ] == obj$has_openmp()))
  if (obj$has_openmp()) {
    expect_equal(sum(res["threadnum", ] == 0), 10)
  } else {
    expect_equal(sum(res["threadnum", ] == -1), 10)
  }
})

test_that("Can get state when multi-threaded", {
  ex <- example_logistic()
  np <- 5
  mod_threaded <- ex$generator$new(ex$pars, 0, np, n_threads = 4)
  mod_threaded$set_index(c(1, 2, 3))
  res_threaded <- mod_threaded$run(1)
  state_threaded <- mod_threaded$state()
  partial_state_threaded <- mod_threaded$state(1L)

  mod_single <- ex$generator$new(ex$pars, 0, np)
  mod_single$set_index(c(1, 2, 3))
  res_single <- mod_single$run(1)
  state_single <- mod_single$state()
  partial_state_single <- mod_single$state(1L)

  expect_equal(res_threaded, res_single)
  expect_equal(state_threaded, state_single)
  expect_equal(partial_state_threaded, partial_state_single)
})

test_that("Can update state when multi-threaded", {
  ex <- example_logistic()
  np <- 3
  mod <- ex$generator$new(ex$pars, 0, np, n_threads = 1)
  y <- mod$run(3)

  new_pars <- list(r = c(0.5, 0.7), K = c(200, 200))

  # update all particles to have the same state
  mod$update_state(pars = new_pars, time = 2, state = c(1, 2))

  expect_equal(mod$state(), matrix(rep(1:3, np), nrow = 3))
  expect_equal(mod$time(), 2)
  expect_equal(mod$pars(), new_pars)

  # update with different state for each particle
  new_state <- matrix(as.double(1:6), nrow = 2)
  expected_state <- matrix(as.double(1:6), nrow = 2)

  mod$update_state(pars = new_pars, time = 2,
                            state = new_state)
  expect_equal(mod$state(), matrix(c(1, 2, 3, 3, 4, 7, 5, 6, 11), nrow = 3))
})

test_that("Can change the number of threads after initialisation", {
  ex <- example_logistic()
  np <- 5
  mod <- ex$generator$new(ex$pars, 0, np)
  expect_equal(withVisible(mod$set_n_threads(2)),
               list(value = 1L, visible = FALSE))
  expect_equal(mod$n_threads(), 2L)
  expect_equal(withVisible(mod$set_n_threads(1)),
               list(value = 2L, visible = FALSE))
})

test_that("Can't change to an impossible thread count", {
  ex <- example_logistic()
  np <- 5
  mod <- ex$generator$new(ex$pars, 0, np)
  expect_error(mod$set_n_threads(0),
               "'n_threads' must be positive")
  expect_error(mod$set_n_threads(-1),
               "'n_threads' must be positive")
})

test_that("Can get openmp support", {
  ex <- example_logistic()
  mod <- ex$generator$new(ex$pars, 0, 5)
  expect_equal(mod$has_openmp(), dust::dust_openmp_support()$has_openmp)
})


test_that("can get rng state", {
  ex <- example_logistic()
  gen <- ex$generator
  pars <- ex$pars
  np <- 10
  mod <- gen$new(pars, 0, np, seed = 1L)
  rng <- mod$rng_state()
  expect_type(rng, "raw")
  expect_identical(rng, dust::dust_rng$new(1, np + 1)$state())
  expect_identical(mod$rng_state(first_only = TRUE),
                   rng[seq_len(32)])
  expect_identical(mod$rng_state(last_only = TRUE),
                   rng[(np * 32 + 1):((np + 1) * 32)])
  expect_error(
    mod$rng_state(first_only = TRUE, last_only = TRUE),
    "Only one of 'first_only' or 'last_only' may be TRUE")
})


test_that("can set rng state into model", {
  ex <- example_logistic()
  gen <- ex$generator
  pars <- list(r = c(0.1, 0.2), K = c(100, 200), v = 0.1)
  mod1 <- gen$new(pars, 0, 10, seed = 1L)
  mod2 <- gen$new(pars, 0, 10, seed = 2L)
  mod1$set_stochastic_schedule(0:10)
  mod2$set_stochastic_schedule(0:10)
  mod1$set_rng_state(mod2$rng_state())
  y1 <- mod1$run(11)
  y2 <- mod2$run(11)
  expect_identical(y1, y2)
  expect_identical(mod1$rng_state(), mod2$rng_state())
})


test_that("Can get information about steps", {
  ex <- example_logistic()
  gen <- ex$generator
  pars <- list(r = c(0.1, 0.2), K = c(100, 200), v = 0.5)
  n_particles <- 5L
  control <- dust_ode_control(debug_record_step_times = TRUE)
  mod <- gen$new(pars, 0L, n_particles, ode_control = control, seed = 1L)
  stats <- mod$ode_statistics()
  schedule <- seq(0, 5, length.out = 11)
  mod$set_stochastic_schedule(schedule)

  ## As usual:
  expect_equal(dim(stats), c(3, n_particles))
  expect_equal(row.names(stats),
               c("n_steps", "n_steps_accepted", "n_steps_rejected"))
  expect_s3_class(stats, "ode_statistics")
  expect_true(all(stats == 0))

  ## But we also have this:
  expect_equal(attr(stats, "step_times"), rep(list(numeric(0)), n_particles))

  mod$run(10)

  stats <- mod$ode_statistics()
  steps <- attr(stats, "step_times")
  expect_equal(lengths(steps), stats["n_steps_accepted", ])
  ## Only end points of the steps are included:
  expect_false(0 %in% steps)
  ## But every point in the stochastic schedule is required:
  all(vapply(steps, function(s) all(schedule[-1] %in% s), TRUE))
})


test_that("information about steps survives shuffle", {
  ex <- example_logistic()
  gen <- ex$generator
  pars <- list(r = c(0.1, 0.2), K = c(100, 200), v = 0.5)
  n_particles <- 5L
  control <- dust_ode_control(debug_record_step_times = TRUE)

  ## First, run through in one go:
  mod <- gen$new(pars, 0L, n_particles, ode_control = control, seed = 1L)
  schedule <- seq(0, 5, length.out = 11)
  mod$set_stochastic_schedule(schedule)
  y1 <- mod$run(10)
  stats1 <- mod$ode_statistics()
  steps1 <- attr(stats1, "step_times")

  ## At reverse, things look ok
  reverse <- rev(seq_len(n_particles))
  mod$reorder(reverse)
  expect_equal(mod$state(), y1[, reverse])
  expect_equal(mod$ode_statistics()[, ], stats1[, reverse])
  expect_equal(attr(mod$ode_statistics(), "step_times"), steps1[reverse])

  ## Then again, but shuffle at half time
  mod <- gen$new(pars, 0L, n_particles, ode_control = control, seed = 1L)
  schedule <- seq(0, 5, length.out = 11)
  mod$set_stochastic_schedule(schedule)
  mod$run(5) # must be part of the stochastic updates
  mod$reorder(reverse)
  ## Reverse the rng state too
  r <- matrix(mod$rng_state(), ncol = n_particles + 1)
  mod$set_rng_state(c(r[, c(reverse, n_particles + 1)]))
  y2 <- mod$run(10)
  stats2 <- mod$ode_statistics()
  steps2 <- attr(stats2, "step_times")

  expect_equal(mod$state(), y1[, reverse])
  expect_equal(mod$ode_statistics()[, ], stats1[, reverse])
  expect_equal(attr(mod$ode_statistics(), "step_times"), steps1[reverse])
})


test_that("can simulate a time series", {
  ex <- example_logistic()
  n_particles <- 5L
  mod <- ex$generator$new(ex$pars, 0, n_particles)
  t <- as.numeric(0:10)
  m <- mod$simulate(t)
  expect_equal(dim(m), c(3, n_particles, length(t)))

  cmp <- ex$generator$new(ex$pars, 0, n_particles)
  for (i in seq_along(t)) {
    expect_identical(m[, , i], cmp$run(t[i]))
  }
})


test_that("can set an index and reflect that in simulate output", {
  ex <- example_logistic()
  n_particles <- 5L
  mod <- ex$generator$new(ex$pars, 0, n_particles)
  mod$set_index(c(n2 = 2, output = 3))
  t <- as.numeric(0:10)
  m <- mod$simulate(t)
  expect_equal(rownames(m), c("n2", "output"))
  expect_equal(dim(m), c(2, n_particles, length(t)))

  ## Same as the full output:
  mod$update_state(time = 0, pars = ex$pars, set_initial_state = TRUE)
  mod$set_index(NULL)
  expect_identical(unname(m), mod$simulate(t)[2:3, , ])
})


test_that("check that simulate times are reasonable", {
  ex <- example_logistic()
  n_particles <- 5L
  mod <- ex$generator$new(ex$pars, 0, n_particles)

  expect_error(
    mod$simulate(seq(-5, 5, 1)),
    "'time_end[1]' must be at least 0", fixed = TRUE)
  expect_error(
    mod$simulate(numeric(0)),
    "'time_end' must have at least one element", fixed = TRUE)
  expect_error(
    mod$simulate(c(0, 1, 2, 3, 2, 5)),
    "'time_end' must be non-decreasing (error on element 5)", fixed = TRUE)
  expect_error(
    mod$simulate(NULL),
    "Expected a numeric vector for 'time_end'", fixed = TRUE)
})


test_that("Can save a model and reload it after repair", {
  skip_if_not_installed("callr")
  ## We need to recompile a model here from scratch; I can't remember
  ## if this is best if it's one that is not in the package though?
  path <- dust_file("examples/ode/logistic.cpp")
  gen <- dust(path, quiet = TRUE)

  tmp_rds <- tempfile()
  suppressWarnings(saveRDS(gen, tmp_rds))

  pars <- list(r = c(0.1, 0.2), K = c(100, 100))

  ## Fails to load because the package environment is not present, and
  ## so can't find the alloc function
  expect_error(callr::r(function(path, pars) {
    gen <- readRDS(path)
    gen$new(pars, 0, 1, seed = 1)$run(10)
  }, list(tmp_rds, pars)), "dust_ode_logistic_alloc")

  ## If we repair the environment it works fine though
  res <- callr::r(function(path, pars) {
    gen <- readRDS(path)
    dust::dust_repair_environment(gen)
    gen$new(pars, 0, 1, seed = 1)$run(10)
  }, list(tmp_rds, pars))

  cmp <- gen$new(pars, 0, 1, seed = 1)$run(10)
  expect_equal(res, cmp)
})


test_that("prevent use of gpu", {
  ex <- example_logistic()
  expect_error(
    ex$generator$new(ex$pars, 0, 1, gpu_config = 1),
    "GPU support not enabled for this object")
})


test_that("can run in deterministic mode", {
  ex <- example_logistic()
  np <- 7
  obj <- ex$generator$new(ex$pars, 0, np, seed = 1, deterministic = TRUE)
  rng_state <- obj$rng_state()
  t1 <- 10
  obj$set_stochastic_schedule(seq_len(t1 - 1))
  y <- obj$run(t1)
  cmp <- logistic_analytic(ex$pars$r, ex$pars$K, t1, c(1, 1))
  expect_equal(obj$rng_state(), rng_state)
  expect_equal(y[1, ], rep(cmp[[1]], np), tolerance = 1e-6)
  expect_equal(y[2, ], rep(cmp[[2]], np), tolerance = 1e-6)
  expect_equal(y[3, ], colSums(y[1:2, ]))
})


test_that("investigate model capabilities", {
  ex <- example_logistic()
  mod <- ex$generator$new(ex$pars, 0, 1)
  expect_type(mod$has_openmp(), "logical")
  expect_false(mod$has_gpu_support())
  expect_false(mod$has_gpu_support(TRUE))
  expect_false(mod$has_compare())
  expect_equal(mod$real_size(), 64)
  expect_equal(mod$rng_algorithm(), "xoshiro256plus")
})


test_that("dummy data methods error on use", {
  ex <- example_logistic()
  n_particles <- 10
  mod <- ex$generator$new(ex$pars, pi, n_particles)
  expect_error(mod$resample(rep(1, n_particles)),
               "Can't yet use resample with continuous-time models")
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


test_that("can retrieve empty params", {
  ex <- example_logistic()
  mod <- ex$generator$new(ex$pars, 0, 10)
  expect_mapequal(mod$param(),
                  list(r = list(required = TRUE),
                       K = list(required = TRUE),
                       v = list(required = FALSE)))
})


test_that("can retrieve empty gpu info", {
  ex <- example_logistic()
  mod <- ex$generator$new(ex$pars, 0, 10)
  ## Another dust model that lacks gpu information; use this as an
  ## expected template:
  expected <- dust::dust_example("sir")$new(list(), 0, 1)$gpu_info()
  expect_equal(mod$gpu_info(), expected)
})


test_that("can set data into models that support it, and compute likelihood", {
  gen <- dust(dust_file("examples/ode/malaria.cpp"), quiet = TRUE)
  d <- dust_data(read.csv(dust_file("extdata/malaria_cases.csv")), "t")
  mod <- gen$new(list(), 0, 1)
  expect_true(mod$has_compare())
  mod$set_data(d)
  mod$set_index(c(Ih = 2))
  for (el in d[1:5]) {
    y <- mod$run(el[[1]])
    cmp <- dbinom(el[[2]]$positive, el[[2]]$tested, y, TRUE)
    expect_equal(mod$compare_data(), cmp)
  }
})


test_that("can resample from ode models", {
  res <- dust_example("logistic")
  np <- 17
  obj <- res$new(list(r = c(0.1, 0.2), K = c(100, 200)), 0, np, seed = 1L)
  s <- obj$state()[1:2, ]
  s[] <- runif(2 * np, 1, 10)
  obj$update_state(state = s)

  rng <- dust_rng$new(obj$rng_state(last_only = TRUE))
  u <- rng$random_real(1)
  w <- runif(obj$n_particles())

  idx <- obj$resample(w)

  expect_equal(idx, resample_index(w, u))
  expect_equal(obj$state()[1:2, ], s[, idx])

  obj2 <- res$new(list(r = c(0.1, 0.2), K = c(100, 200)), 0, np, seed = 1L)
  obj2$update_state(state = s[, idx])

  expect_identical(obj$run(50), obj2$run(50))
})
