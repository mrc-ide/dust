test_that("can construct an ode control object", {
  control <- dust_ode_control()
  expect_s3_class(control, "dust_ode_control")
  expect_setequal(
    names(control),
    c("max_steps", "atol", "rtol", "step_size_min", "step_size_max",
      "debug_record_step_times"))
  expect_true(all(vapply(control, is.null, TRUE)))
})


test_that("can set values into an ode control object", {
  control <- dust_ode_control(max_steps = 1e4, atol = 1e-3, rtol = 1e-6,
                              step_size_min = 1e-8, step_size_max = Inf,
                              debug_record_step_times = FALSE)
  expect_s3_class(control, "dust_ode_control")
  expect_mapequal(
    control,
    list(max_steps = 1e4, atol = 1e-3, rtol = 1e-6,
         step_size_min = 1e-8, step_size_max = Inf,
         debug_record_step_times = FALSE))
})


## It's likely that we would be better served doing this validation at
## the point of construction really.
test_that("Can validate ode control", {
  ex <- example_logistic()
  ## Not a great error message, due to the non-scalar return value of
  ## as_integer
  expect_error(
    ex$generator$new(ex$pars, 10, 1,
                     ode_control = dust_ode_control(max_steps = pi)),
    "All elements of 'max_steps' must be integer-like")
  expect_error(
    ex$generator$new(ex$pars, 10, 1,
                     ode_control = dust_ode_control(max_steps = integer())),
    "Expected 'max_steps' to be a scalar value")
  expect_error(
    ex$generator$new(ex$pars, 10, 1,
                     ode_control = dust_ode_control(atol = numeric())),
    "Expected 'atol' to be a scalar value")
  expect_error(
    ex$generator$new(ex$pars, 10, 1,
                     ode_control = dust_ode_control(
                       debug_record_step_times = logical())),
    "'debug_record_step_times' must be a non-missing scalar logical")
})
