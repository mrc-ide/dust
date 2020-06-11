dust <- function(alloc, run, reset, classname = "dust") {
  R6::R6Class(
    classname,
    cloneable = FALSE,

    private = list(
      cpp_alloc = alloc,
      cpp_run = run,
      cpp_reset = reset,
      ptr = NULL
    ),

    public = list(
      initialize = function(data, step, n_particles, seed) {
        private$ptr <- .Call(private$cpp_alloc, data, step, n_particles, seed)
      },

      run = function(step_end) {
        .Call(private$cpp_run, private$ptr, step_end)
      },

      reset = function(data, step) {
        .Call(private$cpp_reset, private$ptr, data, step)
      }
    ))
}


## Avoid note about "private" not being found and inspection of .Call
## problems; these symbols will be filled in correctly by the factory
## above.
private <- list(cpp_alloc = structure(list(), class = "NativeSymbolInfo"),
                cpp_run = structure(list(), class = "NativeSymbolInfo"),
                cpp_reset = structure(list(), class = "NativeSymbolInfo"))
