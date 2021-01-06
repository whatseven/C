cmake_minimum_required (VERSION 3.10)
PROJECT(generate_data_1216)

FILE(GLOB SOURCE_FILE *.cpp *.h)

add_executable(${PROJECT_NAME} ${SOURCE_FILE})
target_include_directories(${PROJECT_NAME} PUBLIC ../../include)
link_general(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} general C)
link_airsim(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} general tinyxml2::tinyxml2)