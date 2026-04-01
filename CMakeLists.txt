# Setup Project
cmake_minimum_required(VERSION 3.16)
project(mg-miniapp CXX)

# Set project vars
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(FetchContent)



FetchContent_Declare(
    ippl
    GIT_REPOSITORY https://github.com/NickvonWit/ippl.git
    GIT_TAG        master # or main, depending on your default branch
)

FetchContent_MakeAvailable(ippl)

# find_package(Kokkos REQUIRED)

add_executable(mg3d_serial miniapp_multigrid_preconditioner_poisson3d.cpp)

add_executable(mg3d_ippl mg3d_ippl.cpp)
target_link_libraries(mg3d_ippl ippl Kokkos::kokkos)

# add_executable(mg3d_kokkos mg3d_kokkos.cpp)
# target_link_libraries(mg3d_kokkos Kokkos::kokkos ippl)

add_executable(mg3d_kokkos_ai mg3d_kokkos_AI.cpp)
target_link_libraries(mg3d_kokkos_ai Kokkos::kokkos)
