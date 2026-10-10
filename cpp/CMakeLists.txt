cmake_minimum_required(VERSION 3.16)
project(fcw_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 실행 시간 비교용이므로 기본 빌드는 Release(최적화)로 설정
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn)

# 본 프로그램
add_executable(fcw main_code.cpp)
# 결과 동일성 검증 프로그램 (main_code.cpp의 위험도 로직을 그대로 포함)
add_executable(verify verify_cpp.cpp)

foreach(target fcw verify)
  target_include_directories(${target} PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
  # 소스의 한글 주석을 MSVC가 UTF-8로 읽도록 설정
  if(MSVC)
    target_compile_options(${target} PRIVATE /utf-8)
  endif()
endforeach()