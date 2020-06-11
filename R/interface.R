dust <- function(alloc, run, classname = "dust") {
  R6::R6Class(
    classname,
    cloneable = FALSE,

    private = list(
      cpp_alloc = alloc,
      cpp_run = run,
      ptr = NULL
    ),

    public = list(
      initialize = function(data, n_particles, seed) {
        private$ptr <- .Call(private$cpp_alloc, data, n_particles, seed)
      },

      run = function(step_end) {
        .Call(private$cpp_run, private$ptr, step_end)
      }
    ))
}


## Avoid note about "private" not being found and inspection of .Call
## problems; these symbols will be filled in correctly by the factory
## above.
private <- list(cpp_alloc = structure(list(), class = "NativeSymbolInfo"),
                cpp_run = structure(list(), class = "NativeSymbolInfo"))
