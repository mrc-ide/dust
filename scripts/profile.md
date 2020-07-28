# Profiling notes

## Modules

```
module load cuda nvcompilers
module load Core/gcc/8.4.0 gcc/8.4.0/openjdk/11.0.2 gcc/8.4.0/r/4.0.2
```

## Getting source

```
git clone https://github.com/mrc-ide/dust
git -C dust checkout device-select-package
```

## Installing dependencies

From within the `dust` directory

```
./scripts/update_cub
./scripts/install_deps
```

## Running the profiles


### System/Timeline volatility (~30s including installation)

```
./scripts/profile_system volatility
```

### System/Timeline sirs (~30s including installation)

```
./scripts/profile_system sirs
```

### Kernel/Compute volatility (~30s including installation)

```
./scripts/profile_compute volatility
```

### Kernel/Compute sirs (~7 mins including installation)

```
./scripts/profile_compute sirs
```

For all of these, if the packages do not need updating then

```
DUST_NO_INSTALL=true ./scripts/profile_system volatility
```
