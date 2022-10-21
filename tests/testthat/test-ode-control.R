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
