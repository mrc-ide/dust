#!/usr/bin/env bash
set -e

mkdir -p bin out src
cat targets | while read target
do
    file="${target}.c"
    src="src/${file}"
    if [[ ! -f $src ]]; then
        echo "Downloading $src"
        (cd src; curl -sSLO "https://prng.di.unimi.it/${file}")
    fi
    exe="bin/$target"
    dest="out/$target"
    def=$(sed -E 's/(xo.*[0-9]+).*/\U\1/' <<< $target)
    echo "Compiling $exe"
    g++ -include $src -D$def -o$exe harness.cpp
    echo "Generating $dest"
    ./$exe > $dest
done

mkdir -p ../tests/testthat/xoshiro-ref
cp -r out/* ../tests/testthat/xoshiro-ref
