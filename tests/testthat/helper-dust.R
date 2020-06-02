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
