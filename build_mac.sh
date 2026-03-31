#!/bin/bash

# Find the Homebrew LLVM installation path
LLVM_PREFIX=$(brew --prefix llvm)

# Set the C and C++ compilers for CMake
export CC="$LLVM_PREFIX/bin/clang"
export CXX="$LLVM_PREFIX/bin/clang++"

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

cp ./build/compile_commands.json .
