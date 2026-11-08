# qwen4exp_x3 -- qwen4exp_fp8's plugin under the ("qwen4exp", "exl3") selection key: the same source (included whole,
# its exports off), answering the "exl3" quantisation descriptor that rad-convert writes for an EXL3 checkpoint
# (xeon's qwen3.8-flash-next-exl3.rad). Built in radiance's tree like qwen4exp_fp8.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(qwen4exp_x3 LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

rad_add_plugin(qwen4exp_x3 architectures SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/qwen4exp_x3.cpp)
