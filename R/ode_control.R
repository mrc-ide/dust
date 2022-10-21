##' Create a control object for controlling the adaptive stepper for
##' systems of ordinary differential equations (ODEs). The returned
##' object can be passed into a continuous-time dust model on
##' initialisation.
##'
##' @title Create a dust_ode_control object.
##'
##' @param max_steps Maxmimum number of steps to take. If the
##'   integration attempts to take more steps that this, it will
##'   throw an error, stopping the integration.
##'
##' @param rtol The per-step relative tolerance.  The total accuracy
##'   will be less than this.
##'
##' @param atol The per-step absolute tolerance.
##'
##' @param step_size_min The minimum step size.  The actual minimum
##'   used will be the largest of the absolute value of this
##'   \code{step_size_min} or \code{.Machine$double.eps}. If the
##'   integration attempts to make a step smaller than this, it will
##'   throw an error, stopping the integration.
##'
##' @param step_size_max The largest step size.  By default there is
##'   no maximum step size (Inf) so the solver can take as large a
##'   step as it wants to.  If you have short-lived fluctuations in
##'   your rhs that the solver may skip over by accident, then specify
##'   a smaller maximum step size here.
##'
##' @param debug_record_step_times Logical, indicating if we should record
##'   the steps taken. This information will be available as part of
##'   the `statistics()` output
##'
##' @export
##' @return A named list of class "dust_ode_control"
dust_ode_control <- function(max_steps = NULL, atol = NULL, rtol = NULL,
                             step_size_min = NULL, step_size_max = NULL,
                             debug_record_step_times = NULL) {
  ctl <- list(max_steps = max_steps, atol = atol, rtol = rtol,
              step_size_min = step_size_min, step_size_max = step_size_max,
              debug_record_step_times = debug_record_step_times)
  class(ctl) <- "dust_ode_control"
  ctl
}
