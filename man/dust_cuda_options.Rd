% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cuda.R
\name{dust_cuda_options}
\alias{dust_cuda_options}
\title{Create CUDA options}
\usage{
dust_cuda_options(
  ...,
  debug = FALSE,
  profile = FALSE,
  fast_math = FALSE,
  flags = NULL
)
}
\arguments{
\item{...}{Arguments passed to \code{\link[=dust_cuda_configuration]{dust_cuda_configuration()}}}

\item{debug}{Logical, indicating if we should compile for debug
(adding \code{-g}, \code{-G} and \code{-O0})}

\item{profile}{Logical, indicating if we should enable profiling}

\item{fast_math}{Logical, indicating if we should enable "fast
maths", which lets the optimiser enable optimisations that break
IEEE compliance and disables some error checking (see \href{https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html}{the CUDA docs}
for more details).}

\item{flags}{Optional extra arguments to pass to nvcc. These
options will not be passed to your normal C++ compiler, nor the
linker (for that use R's user Makevars system). This can be used
to do things like tune the maximum number of registers
(\verb{--maxrregcount x}). If not \code{NULL}, this must be a character
vector, which will be concatenated with spaces between options.}
}
\value{
An object of type \code{cuda_options}, which can be passed into
\link{dust} as argument \code{gpu}
}
\description{
Create options for compiling for CUDA.  Unless you need to change
paths to libraries/headers, or change the debug level you will
probably not need to directly use this. However, it's potentially
useful to see what is being passed to the compiler.
}
\examples{
tryCatch(
  dust::dust_cuda_options(),
  error = function(e) NULL)
}
\seealso{
\link{dust_cuda_configuration} which identifies and
returns the core CUDA configuration (often used implicitly by
this function).
}
