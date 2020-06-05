cache <- new.env(parent = emptyenv())
example_sir <- function() {
  if (is.null(cache$sir)) {
    sir <- odin:::compile("example/sir.c") # nolint
    dyn.load(sir$dll)
    sir$model <- dust_model("sir2_create", "sir2_update", "sir2_free",
                            sir$base)
    cache$sir <- sir
  }
  cache$sir$model
}

example_walk <- function() {
  if (is.null(cache$walk)) {
    walk <- odin:::compile("example/walk.c") # nolint
    dyn.load(walk$dll)
    walk$model <- dust_model("walk_create", "walk_update", "walk_free",
                             walk$base)
    cache$walk <- walk
  }
  cache$walk$model
}

helper_run_dust <- function(n, by, obj) {
  res <- vector("list", n)
  for (i in seq_len(n)) {
    if (i == 1) {
      value <- NULL
    } else {
      value <- matrix(dust_run(obj, (i - 1) * by))
    }
    res[[i]] <- list(step = dust_step(obj),
                     state = matrix(dust_state(obj)),
                     value = value)
  }
  res
}
