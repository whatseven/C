cmake_minimum_required (VERSION 3.10)
PROJECT(C)

set(is_build_model true)

set(YILIN 1)

IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(AIRSIM_ROOT "/root/code/AirSim")
ELSEIF(YILIN)
    set(AIRSIM_ROOT "d:\\env\\AirSim")
    set (CMAKE_EXE_LINKER_FLAGS "/incremental ${CMAKE_EXE_LINKER_FLAGS}")
ELSEIF(RUIQI)
    set(AIRSIM_ROOT "F:\\Unreal\\UE_4.24\\AirSim")
    set (CMAKE_EXE_LINKER_FLAGS "/incremental ${CMAKE_EXE_LINKER_FLAGS}")
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")

function (configure_airsim)
    INCLUDE_DIRECTORIES("${AIRSIM_ROOT}/AirLib/include")
endfunction ()

function (link_airsim arg)
    IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
        message(${ARGV0})
        target_link_libraries(${ARGV0} "/root/code/AirSim/AirLib/lib/libAirLib.a")
        target_link_libraries(${ARGV0} general "/root/code/AirSim/AirLib/deps/rpclib/lib/librpc.a")
    ELSE ()
        target_link_libraries(${ARGV0} debug "${AIRSIM_ROOT}/AirLib/deps/rpclib/lib/x64/Debug/rpc.lib")
        target_link_libraries(${ARGV0} debug "${AIRSIM_ROOT}/AirLib/lib/x64/Debug/AirLib.lib")
        target_link_libraries(${ARGV0} optimized "${AIRSIM_ROOT}/AirLib/lib/x64/Release/AirLib.lib")
        target_link_libraries(${ARGV0} optimized "${AIRSIM_ROOT}/AirLib/deps/rpclib/lib/x64/Release/rpc.lib")
    ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")
endfunction ()

if(${is_build_model})
    
endif(${is_build_model})

find_package(Eigen3 REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(argparse CONFIG REQUIRED)
find_package(CGAL CONFIG REQUIRED)
find_package(Boost REQUIRED filesystem)
find_package(OpenMP REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(tinyxml2 CONFIG REQUIRED)

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
    target_link_libraries(${ARGV0} general tinyxml2::tinyxml2)
endfunction()

FILE(GLOB LIBRARY_FILE "include/*.cpp" "include/*.h")

configure_airsim()
add_library(C ${LIBRARY_FILE})
target_include_directories(C PRIVATE ${Boost_INCLUDE_DIRS})

add_subdirectory(src)

