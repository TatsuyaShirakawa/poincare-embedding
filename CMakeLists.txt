cmake_minimum_required(VERSION 3.1)

project(poincare-embedding)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(poincare_embedding
  ${CMAKE_SOURCE_DIR}/src/poincare_embedding.cpp
  )


# pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(poincare_embedding Threads::Threads)
