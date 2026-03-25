cmake_minimum_required(VERSION 3.16)
project(mg-miniapp CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Kokkos REQUIRED)

add_executable(mg3d_serial miniapp_multigrid_preconditioner_poisson3d.cpp)

add_executable(mg3d_kokkos mg3d_kokkos.cpp)
target_link_libraries(mg3d_kokkos Kokkos::kokkos)

add_executable(mg3d_kokkos_ai mg3d_kokkos_AI.cpp)
target_link_libraries(mg3d_kokkos_ai Kokkos::kokkos)
