test_that("xoshiro output agrees with reference data", {
  path <- "xoshiro-ref"
  status <- list()
  for (name in dir(path)) {
    obj <- dust_rng_pointer$new(seed = 42, algorithm = name)
    res <- test_xoshiro_run(obj)
    cmp <- readLines(sprintf("%s/%s", path, name))
    expect_equal(res, cmp)
  }
})
