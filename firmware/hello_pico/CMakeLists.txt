cmake_minimum_required(VERSION 3.13)

include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(hello_pico C CXX ASM)
set(CMAKE_C_STANDARD 11)

pico_sdk_init()

add_executable(hello_pico main.c)

pico_add_extra_outputs(hello_pico)
target_link_libraries(hello_pico pico_stdlib)
pico_enable_stdio_usb(hello_pico 1)
pico_enable_stdio_uart(hello_pico 0)
