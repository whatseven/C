cmake_minimum_required (VERSION 3.10)
PROJECT(C)

set(is_build_model true)

IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
ELSE ()
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")

if(${is_build_model})
    
endif(${is_build_model})

find_package(Eigen3 REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(argparse CONFIG REQUIRED)
find_package(CGAL CONFIG REQUIRED)
find_package(Boost REQUIRED filesystem)
find_package(OpenMP REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

function(link_general arg)
    target_include_directories(${ARGV0} PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(${ARGV0} general  ${OpenCV_LIBRARIES})
    target_link_libraries(${ARGV0} ${Boost_LIBRARIES})
    target_link_libraries(${ARGV0} general argparse::argparse)
    target_link_libraries(${ARGV0} general CGAL::CGAL)
    target_link_libraries(${ARGV0} general tinyobjloader::tinyobjloader)
endfunction()

FILE(GLOB LIBRARY_FILE "include/*.cpp" "include/*.h")

add_library(C ${LIBRARY_FILE})
target_include_directories(C PRIVATE ${Boost_INCLUDE_DIRS})

add_subdirectory(src)

