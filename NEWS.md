# dust 0.3.0

* Overhaul of the RNG interface, where we now pass state and use free functions; this is needed to support future development work (#60)

# dust 0.2.2

* New `$rng_state()` method for getting the RNG state as a raw vector

# dust 0.2.0

* Use cpp11 as the backend (#22)

# dust 0.1.5

* Simpler RNG interface; we now always use as many RNG streams as there are particles (#51)

# dust 0.1.4

* New function `dust::dust_simulate` which provides a helper for running a simulation while collecting output (#7)

# dust 0.1.3

* Allow `$set_state()` to accept an initial state of `NULL`, in which case only the time is set (#48)

# dust 0.1.2

* Allow `$set_state()` to accept an initial step too. This can be a vector if particles start at different initial steps, in which case all particles are run up to the latest step (#45)

# dust 0.1.1

* Allow `$set_state()` to accept a matrix in order to start particles with different starting values (#43)
