find_package(GTest REQUIRED)

enable_testing()

set(link_lib glog::glog GTest::gtest)

add_executable(infer_test main_test.cpp tensor_test.cpp runtime_attr_test.cpp runtime_ir_test.cpp runtime_param_test.cpp)

target_link_libraries(infer_test ${link_lib} ${link_math_lib})
target_link_directories(infer_test PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(infer_test infer)

include(GoogleTest)
gtest_discover_tests(infer_test)

