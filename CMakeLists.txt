cmake_minimum_required(VERSION 3.22.1)
project(Direction_Vol_C90 C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS OFF)

include_directories(include include/direction)

set(SOURCES
    src/utils.c
    src/data.c
    src/lstm.c
    src/model.c
    src/params.c
    src/vol.c
    src/train.c
    src/backtest.c
    src/metrics.c
    src/walkfwd.c
    src/direction/data_direction.c
    src/direction/predict_direction.c
    main.c
)

add_executable(Direction_Vol_C90 ${SOURCES})
target_link_libraries(Direction_Vol_C90 $<$<C_COMPILER_ID:GNU,Clang>:m>)
target_compile_options(Direction_Vol_C90 PRIVATE $<$<C_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic> $<$<C_COMPILER_ID:MSVC>:/W4>)

# Unified test runner — all debug tests in one executable
set(TEST_ALL_SOURCES
    src/utils.c
    src/data.c
    src/lstm.c
    src/model.c
    src/params.c
    src/vol.c
    src/train.c
    src/backtest.c
    src/metrics.c
    src/walkfwd.c
    src/direction/data_direction.c
    src/direction/predict_direction.c
    tests/test_all_main.c
    tests/test_td1.c
    tests/test_td2.c
    tests/test_td3.c
    tests/test_td4.c
    tests/test_tm1.c
    tests/test_tm2.c
    tests/test_tf1.c
    tests/test_tf2.c
    tests/test_tf3.c
    tests/test_tf4.c
    tests/test_ts1.c
    tests/test_ts2.c
    tests/test_ts3.c
    tests/test_ts4.c
    tests/test_tn1.c
    tests/test_tn2.c
    tests/test_tn3.c
    tests/test_tv1.c
    tests/test_tv2.c
    tests/test_tv3.c
    tests/test_tl1.c
    tests/test_tl2.c
    tests/test_tl3.c
    tests/test_tl4.c
    tests/test_tl5.c
    tests/test_tb1.c
    tests/test_tb2.c
    tests/test_tb3.c
    tests/test_tb4.c
    tests/test_tb4b.c
    tests/test_tb5.c
    tests/test_tb6.c
    tests/test_te1.c
    tests/test_te2.c
    tests/test_te3.c
    tests/test_tw1.c
    tests/test_tw2.c
    tests/test_tw3.c
    tests/test_tg1.c
    tests/test_tg2.c
    tests/test_tg3.c
)

add_executable(test_all ${TEST_ALL_SOURCES})
target_link_libraries(test_all $<$<C_COMPILER_ID:GNU,Clang>:m>)
target_compile_options(test_all PRIVATE $<$<C_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic> $<$<C_COMPILER_ID:MSVC>:/W4>)
