#!/bin/bash

spack load /ax27afe
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

