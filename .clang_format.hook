#!/bin/bash
set -e

readonly VERSION="13.0.0"

version=$(clang-format -version)

if ! [[ $(python -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1$2}') -ge 36 ]]; then
    echo "clang-format installation by pip need python version great equal 3.6,
          please change the default python to higher version."
    exit 1
fi

if ! [[ $version == *"$VERSION"* ]]; then
    # low version of pip may not have the source of clang-format whl
    pip install --upgrade pip
    pip install clang-format==13.0.0
fi

clang-format $@
