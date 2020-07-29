main <- function(dest) {
  ## Check out each commit, then run:
  vol_c <- dust::dust("inst/examples/volatility.cpp")
  vol_g <- gpupkg:::volatility

  test <- function(gen, n_particles, n_steps, n_threads = 1L) {
    gen$new(list(), 0, n_particles, n_threads)$run(n_steps)
  }

  n_particles <- 2^(0:16)
  n_steps <- c(10, 100, 1000)

  res_c <- bench::press(
    n_particles = n_particles,
    n_steps = n_steps,
    model = "volatility",
    on = "cpu",
    bench::mark(test(vol_c, n_particles, n_steps)))
  res_g <- bench::press(
    n_particles = n_particles,
    n_steps = n_steps,
    model = "volatility",
    on = "gpu",
    bench::mark(test(vol_g, n_particles, n_steps)))

  res <- rbind(res_c, res_g)
  res$expression <- NULL
  res$result <- NULL
  res$gc <- NULL
  res$time <- NULL
  dir.create(dirname(dest), FALSE, TRUE)
  saveRDS(res, dest)
}

main(commandArgs(TRUE)[[1]])
