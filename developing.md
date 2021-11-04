## Developing dust

Adding new methods to the dust object requires a few steps (most of which will be caught by CI)

* If you're doing something complicated, you'll find it more pleasant to delete all example .cpp files from `src/` (`sir.cpp`, `sirs.cpp`, `variable.cpp`, `volatility.cpp`, `walk.cpp`) particularly if your changes break compilation. Then you'll be in a position to iterate faster and control where the compilation errors are thrown (package tests will fail with this change, but the package will work fine).
* If you have made any change to the dust class (updating a method, adding an argument etc) you must regenerate the examples by running `./scripts/update_example` - be sure to edit *both* the `.cpp` and `.hpp` files or you will get compilation errors as the definitions will not match.
* If you make any changes to the dust class interface (updating a method, adding an argument or changing the documentation) you must run `./scripts/update_dust_generator` before running `devtools::document()`. Running `make roxygen` will do this for you
* The cuda vignette is built offline because it requires access to a CUDA toolchain and compatible device.  The script `./scripts/build_cuda_vignette` will update the version in the package

All PRs, no matter small, must increase the version number. This is enforced by github actions. We aim to use [semanitic versioning](https://semver.org/) as much as is reasonable, but our main aim is that all commits to master are easily findable in future.

## Random number library

We keep the random number library in `inst/include/dust` so that it does not depend on anything else in the source tree so that it could be reused in other projects (R or otherwise).

To update the underlying generator code with the reference implementations at https://prng.di.unimi.it/ you should run the script at `./extra/generate.sh` which will download, compile, and run small programs with the upstream implementation and write out reference output in the test directory.

## Headers

As the project has become more complex, keeping the headers under control has become harder. Basic principles here:

* Every header must be self-contained (in that it can be included in any order and pulls in all its dependencies)
* Includes should be grouped by (1) System (stdlib, omp etc), (2) cpp11 (if an interface file), (3) dust's includes. Each block should be alphabetical - any deviations required to support something compiled correctly should be removed.  There are exceptions to this: for example `random/random.hpp` imports all distributions alphabetically separately from the generators so that it's clearer.
* Only include cpp11 (or R) files from files within `dust/interface/`; within these files throw errors only with `throw`, not with `cpp11::stop` (these errors will be correctly caught).

The script `scripts/check_headers` will validate that headers are self contained and that only interface headers include `cpp11` files (directly or indirectly).
