cmake_minimum_required (VERSION 3.10)
PROJECT(capture_airsim_photo)

FILE(GLOB SOURCE_FILE *.cpp *.h)
find_package(jsoncpp CONFIG REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILE})
target_include_directories(${PROJECT_NAME} PUBLIC ../../include)
target_include_directories(${PROJECT_NAME} PUBLIC ../sig20/main)
link_general(${PROJECT_NAME})
link_airsim(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} general C)
target_link_libraries(${PROJECT_NAME} jsoncpp_lib)
