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

## Updating the package sources

Run

```
./scripts/update_gpupkg
```

after changing the example files or Makevars

## Running the profiles

Run

```
./scripts/update_for_profile
```

### System/Timeline volatility (~15s)

```
./scripts/profile_system volatility
```

### System/Timeline sirs (~15s)

```
./scripts/profile_system sirs
```

### Kernel/Compute volatility (~15s)

```
./scripts/profile_compute volatility
```

### Kernel/Compute sirs (~7 mins)

```
./scripts/profile_compute sirs
```
