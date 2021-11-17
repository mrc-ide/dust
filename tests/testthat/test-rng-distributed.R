test_that("Can create a set of distributed random number states", {
  s <- dust_rng_distributed_state(1L, n_nodes = 2)
  expect_type(s, "list")
  expect_length(s, 2)
  cmp <- dust_rng$new(1)
  expect_equal(s[[1]], cmp$state())
  cmp$long_jump()
  expect_equal(s[[2]], cmp$state())
})


test_that("Can create a set of distributed random number states", {
  s <- dust_rng_distributed_state(1L, n_streams = 3, n_nodes = 2)
  expect_type(s, "list")
  expect_length(s, 2)
  cmp <- dust_rng$new(1, 3)
  expect_equal(s[[1]], cmp$state())
  cmp$long_jump()
  expect_equal(s[[2]], cmp$state())
})


test_that("can create distributed rng pointers", {
  algorithm <- "xoroshiro128starstar"
  p <- dust_rng_distributed_pointer(1L, n_streams = 3, n_nodes = 2,
                                    algorithm = algorithm)
  expect_type(p, "list")
  expect_length(p, 2)
  expect_equal(
    p[[1]]$state(),
    dust_rng_pointer$new(1, 3, 0, algorithm = algorithm)$state())
  expect_equal(
    p[[2]]$state(),
    dust_rng_pointer$new(1, 3, 1, algorithm = algorithm)$state())
})


test_that("Fetch types from generators", {
  skip_for_compilation()
  algorithm <- "xoshiro128plus"
  res <- dust(dust_file("examples/walk.cpp"), real_type = "float",
              quiet = TRUE)
  expect_equal(check_algorithm(res), algorithm)
  expect_equal(check_algorithm(res$new(list(sd = 1), 0, 10)), algorithm)

  p <- dust_rng_distributed_pointer(1, 3, algorithm = res)
  expect_equal(p[[1]]$algorithm, algorithm)
  expect_equal(
    p[[1]]$state(),
    dust_rng_pointer$new(1, 3, algorithm = algorithm)$state())
})
