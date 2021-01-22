cmake_minimum_required (VERSION 3.10)
PROJECT(main)

FILE(GLOB SOURCE_FILE *.cpp *.h)
find_package(Pangolin CONFIG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILE})
target_include_directories(${PROJECT_NAME} PUBLIC ../../../include)
link_general(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} general C)
link_airsim(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} general pangolin)
target_link_libraries(${PROJECT_NAME} jsoncpp_lib)
