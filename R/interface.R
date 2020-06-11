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
