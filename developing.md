## Developing dust

Adding new methods to the dust object requires a few steps (most of which will be caught by CI)

* If you're doing something complicated, you'll find it more pleasant to delete all example .cpp files from `src/` (`sir.cpp`, `sirs.cpp`, `variable.cpp`, `volatility.cpp`, `walk.cpp`) particularly if your changes break compilation. Then you'll be in a position to iterate faster and control where the compilation errors are thrown (package tests will fail with this change, but the package will work fine).
* If you have made any change to the dust class (updating a method, adding an argument etc) you must regenerate the examples by running `./scripts/update_example`
* If you make any changes to the dust class interface (updating a method, adding an argument or changing the documentation) you must run `./scripts/update_dust_generator` before running `devtools::document()`. Running `make roxygen` will do this for you
* The cuda vignette is built offline because it requires access to a CUDA toolchain and compatible device.  The script `./scripts/build_cuda_vignette` will update the version in the package

All PRs, no matter small, must increase the version number. This is enforced by github actions. We aim to use [semanitic versioning](https://semver.org/) as much as is reasonable, but our main aim is that all commits to master are easily findable in future.

## Random number library

We keep the random number library in `inst/include/dust` so that it does not depend on anything else in the source tree so that it could be reused in other projects (R or otherwise).

To update the underlying generator code with the reference implementations at https://prng.di.unimi.it/ you should run the script at `./extra/generate.sh` which will download, compile, and run small programs with the upstream implementation and write out reference output in the test directory.
