test_that("Can create pointer object", {
  obj <- dust_rng_pointer$new()
  expect_true(obj$is_current())
  expect_type(obj$state(), "raw")
  expect_length(obj$state(), 32)
  expect_equal(obj$algorithm(), "xoshiro256plus")
  r <- obj$state()
  obj$sync()
  expect_identical(r, obj$state())
})


test_that("Using object requires sync", {
  obj <- dust_rng_pointer$new()
  expect_true(obj$is_current())
  r <- obj$state()
  pi_dust(100, obj)
  expect_false(obj$is_current())
  expect_identical(obj$state(), r)
  obj$sync()
  expect_true(obj$is_current())
  expect_false(identical(obj$state(), r))
})


test_that("Invalidated pointers can be rebuilt", {
  obj1 <- dust_rng_pointer$new()
  obj2 <- corrupt_pointer(obj1)
  expect_equal(
    pi_dust(100, obj2),
    pi_dust(100, obj1))

  ## This saves that the pointer is invalid:
  obj3 <- corrupt_pointer(obj2)
  expect_error(
    pi_dust(100, obj3),
    "Can't unserialise an rng pointer that was not synced")

  ## But if we sync things it's ok:
  obj2$sync()
  obj4 <- corrupt_pointer(obj2)
  expect_equal(
    pi_dust(100, obj4),
    pi_dust(100, obj1))
})
