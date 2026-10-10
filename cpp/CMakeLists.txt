cmake_minimum_required(VERSION 3.16)
project(fcw_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 실행 시간 비교용이므로 기본 빌드는 Release(최적화)로 설정
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn)

add_executable(fcw main_code.cpp)
target_include_directories(fcw PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(fcw PRIVATE ${OpenCV_LIBS})

# 소스의 한글 주석을 MSVC가 UTF-8로 읽도록 설정
if(MSVC)
  target_compile_options(fcw PRIVATE /utf-8)
endif()
