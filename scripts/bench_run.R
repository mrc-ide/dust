main <- function(model, dest) {
  ## Check out each commit, then run:
  if (model == "volatility") {
    gen_c <- dust::dust("inst/examples/volatility.cpp")
    gen_g <- gpupkg:::volatility
  } else {
    gen_c <- dust::dust("inst/examples/sirs.cpp")
    gen_g <- gpupkg:::sirst
  }

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
    bench::mark(test(gen_c, n_particles, n_steps)))
  res_g <- bench::press(
    n_particles = n_particles,
    n_steps = n_steps,
    model = "volatility",
    on = "gpu",
    bench::mark(test(gen_g, n_particles, n_steps)))

  res <- rbind(res_c, res_g)
  res$expression <- NULL
  res$result <- NULL
  res$gc <- NULL
  res$time <- NULL
  dir.create(dirname(dest), FALSE, TRUE)
  saveRDS(res, dest)
}

main(commandArgs(TRUE)[[1]], commandArgs(TRUE)[[2]])
