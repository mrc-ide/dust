# Developing dust

As the project has grown, we've added a few processes around updating parts of the package.

## Adding features

Adding new methods to the dust object requires a few steps (most of which will be caught by CI)

* If you're doing something complicated, you'll find it more pleasant to delete all example .cpp files from `src/` (`sir.cpp`, `sirs.cpp`, `variable.cpp`, `volatility.cpp`, `walk.cpp`) particularly if your changes break compilation.  Then you'll be in a position to iterate faster and control where the compilation errors are thrown (package tests will fail with this change, but the package will work fine). You can use `./script/remove_examples` to automate this.
* If you have made any change to the dust class (updating a method, adding an argument etc) you must regenerate the examples by running `./scripts/update_example` - be sure to edit *both* the `.cpp` and `.hpp` files or you will get compilation errors as the definitions will not match.
* If you make any changes to the dust class interface (updating a method, adding an argument or changing the documentation) you must run `./scripts/update_dust_generator` before running `devtools::document()`. Running `make roxygen` will do this for you
* The gpu vignette is built offline because it requires access to a CUDA toolchain and compatible device.  The script `./scripts/build_gpu_vignette` will update the version in the package

## Versioning

All PRs, no matter small, must increase the version number. This is enforced by github actions. We aim to use [semanitic versioning](https://semver.org/) as much as is reasonable, but our main aim is that all commits to master are easily findable in future.

* If you make any change you will need to increase the version number in `DESCRIPTION` before the GitHub actions checks will pass
* If you make any changes in `inst/include/dust` then you also need to run `./scripts/update_version` to reflect this new version number into the header file contents. This is also automatically checked on GitHub actions.

## Random number library

We keep the random number library in `inst/include/dust` so that it does not depend on anything else in the source tree so that it could be reused in other projects (R or otherwise).

To update the underlying generator code with the reference implementations at https://prng.di.unimi.it/ you should run the script at `./extra/generate.sh` which will download, compile, and run small programs with the upstream implementation and write out reference output in the test directory.

## Headers

As the project has become more complex, keeping the headers under control has become harder. Basic principles here:

* Every header must be self-contained (in that it can be included in any order and pulls in all its dependencies)
* Includes should be grouped by (1) System (stdlib, omp etc), (2) cpp11 (if an interface file), (3) dust's includes. Each block should be alphabetical - any deviations required to support something compiled correctly should be removed.  There are exceptions to this: for example `random/random.hpp` imports all distributions alphabetically separately from the generators so that it's clearer.
* Only include cpp11 (or R) files from files within `dust/r/`; within these files throw errors only with `throw`, not with `cpp11::stop` (these errors will be correctly caught).

The script `scripts/check_headers` will validate that headers are self contained and that only interface headers include `cpp11` files (directly or indirectly).

## Documentation

There are lots of places to consider putting documentation

* Following R package standards, all user facing functions must be documented and have examples, we use roxygen for this (generates files in `man/`).
* Vignettes form the backbone of the documentation. Mostof these are directly built as normal (in `vignettes/`) but a few are precomputed (see [`vignettes_src/`](vignette_src)) where they need special resources such as access to a GPU or more cores than usually available.
* The C++ interface is documented using [doxygen](https://www.doxygen.nl/index.html) tags within API functions, and then compiled to html using [sphinx](https://www.sphinx-doc.org/)/[breathe](https://breathe.readthedocs.io/en/latest/).  Running `./scripts/docs_build_cpp` will build these at `sphinx/_build/html` for local development

## Debugging cuda

```
R -d cuda-gdb
```

It might be useful to set this:

```
set cuda api_failures stop
```

Then start the process by running `r <enter>` and all the usual gdb things work reasonably well.

To find memory errors, compile a model with `gpu = dust::dust_cuda_options(debug = TRUE)` to enable debug symbols, then run with

```
R -d cuda-memcheck
```

which will report the location of invalid access.

Using `printf()` within kernels works fine, though it does make a mess of the screen.

## Finding unexpcted double precision code

You want [the `-warn-double-usage`](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options-warn-on-double-precision-use) argument, passed via `-Xptxas`.

```
gpu <- dust::dust_cuda_options(
  fast_math = TRUE, profile = FALSE, quiet = FALSE, debug = FALSE,
  flags = paste("--keep --source-in-ptx --generate-line-info",
                "-Xptxas -warn-double-usage"))
```

The additional flags are required to make this nice to use:

* `--source-in-ptx`: interleaves the source with the ptx so you know where the f64 calls come from
* `--keep`: retains the ptx so that you can read it (otherwise it is deleted)
*` --generate-line-info` is required for the `--source-in-ptx` option to do anything
