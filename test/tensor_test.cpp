#include <glog/logging.h>
#include <iostream>
#include <gtest/gtest.h>
#include "data/tensor.h"

using namespace std;

TEST(test_tensor, tensor_init1) {
  using namespace kuiper_infer;
  Tensor<float> f1(3, 224, 224);
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);
  ASSERT_EQ(f1.size(), 224 * 224 * 3);
}

TEST(test_tensor, tensor_init1_1d) {
  using namespace kuiper_infer;
  Tensor<float> f1(3);
  const auto& raw_shapes = f1.raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 1);
  ASSERT_EQ(raw_shapes.at(0), 3);
}

// TEST(TensorTest, Constructor3Assertions) {
//     vector<float> elements = {1, 2, 3, 4, 5, 6, 7, 8, 9 };
//     vector<uint32_t> shapes = {3, 3};
//     Tensor<float> tensor(elements.data(), shapes);
//     tensor.show();
// }

// TEST(TensorTest, ReshapeAssertions) {
//     Tensor<float> tensor(1, 2, 3);
//     tensor.reshape({2, 3, 1});
//     tensor.show();
// }

// TEST(TensorTest, FlattenAssertions) {
//     Tensor<float> tensor(1, 2, 3);
//     tensor.flatten();
//     tensor.show();
// }
