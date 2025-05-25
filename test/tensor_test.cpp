#include <glog/logging.h>
#include <iostream>
#include <gtest/gtest.h>
#include "data/tensor.h"
#include "data/tensor_util.h"
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

TEST(test_tensor, transform2) {
  using namespace kuiper_infer;

  Tensor<float> f3(3, 3, 3);
  ASSERT_EQ(f3.empty(), false);
  f3.fill(1.f);
  f3.transform([](const float& value) { return value * 2.f; });
  for (int i = 0; i < f3.size(); ++i) {
    ASSERT_EQ(f3.index(i), 2.f);
  }
}

TEST(test_tensor, clone) {
  using namespace kuiper_infer;

  std::shared_ptr<Tensor<float>> f3 = std::make_shared<Tensor<float>>(3, 3, 3);
  ASSERT_EQ(f3->empty(), false);
  f3->randn();

  const auto& f4 = TensorClone(f3);
  assert(f4->data().memptr() != f3->data().memptr());
  ASSERT_EQ(f4->size(), f3->size());
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), f4->index(i));
  }
}

TEST(test_tensor, flatten1) {
  using namespace kuiper_infer;

  Tensor<float> f3(3, 3, 3);
  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back(float(i));
  }
  f3.fill(values);
  f3.flatten(false);
  ASSERT_EQ(f3.channels(), 1);
  ASSERT_EQ(f3.rows(), 1);
  ASSERT_EQ(f3.cols(), 27);
  ASSERT_EQ(f3.index(0), 0);
  ASSERT_EQ(f3.index(1), 3);
  ASSERT_EQ(f3.index(2), 6);

  ASSERT_EQ(f3.index(3), 1);
  ASSERT_EQ(f3.index(4), 4);
  ASSERT_EQ(f3.index(5), 7);

  ASSERT_EQ(f3.index(6), 2);
  ASSERT_EQ(f3.index(7), 5);
  ASSERT_EQ(f3.index(8), 8);
}

TEST(test_tensor, flatten2) {
  using namespace kuiper_infer;

  Tensor<float> f3(3, 3, 3);
  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back(float(i));
  }
  f3.fill(values);
  f3.flatten(true);
  for (int i = 0; i < 27; ++i) {
    ASSERT_EQ(f3.index(i), i);
  }
}

TEST(test_tensor, tensor_broadcast1) {
  using namespace kuiper_infer;
  const std::shared_ptr<ftensor>& tensor1 = TensorCreate<float>({3, 1, 1});
  const std::shared_ptr<ftensor>& tensor2 = TensorCreate<float>({3, 32, 32});

  const auto& [tensor11, tensor21] = TensorBroadcast(tensor1, tensor2);
  ASSERT_EQ(tensor21->channels(), 3);
  ASSERT_EQ(tensor21->rows(), 32);
  ASSERT_EQ(tensor21->cols(), 32);

  ASSERT_EQ(tensor11->channels(), 3);
  ASSERT_EQ(tensor11->rows(), 32);
  ASSERT_EQ(tensor11->cols(), 32);

  ASSERT_TRUE(arma::approx_equal(tensor21->data(), tensor2->data(), "absdiff", 1e-4));
}

TEST(test_tensor, tensor_broadcast2) {
  using namespace kuiper_infer;
  const std::shared_ptr<ftensor>& tensor1 = TensorCreate<float>({3, 32, 32});
  const std::shared_ptr<ftensor>& tensor2 = TensorCreate<float>({3, 1, 1});
  tensor2->randn();

  const auto& [tensor11, tensor21] = TensorBroadcast(tensor1, tensor2);
  ASSERT_EQ(tensor21->channels(), 3);
  ASSERT_EQ(tensor21->rows(), 32);
  ASSERT_EQ(tensor21->cols(), 32);

  ASSERT_EQ(tensor11->channels(), 3);
  ASSERT_EQ(tensor11->rows(), 32);
  ASSERT_EQ(tensor11->cols(), 32);

  for (uint32_t i = 0; i < tensor21->channels(); ++i) {
    float c = tensor2->at(i, 0, 0);
    const auto& in_channel = tensor21->slice(i);
    for (uint32_t j = 0; j < in_channel.size(); ++j) {
      ASSERT_EQ(in_channel.at(j), c);
    }
  }
}

TEST(test_tensor, mul5) {
  using namespace kuiper_infer;
  const auto& f1 = std::make_shared<Tensor<float>>(3, 224, 224);
  f1->fill(3.f);
  const auto& f2 = std::make_shared<Tensor<float>>(3, 1, 1);
  f2->fill(2.f);

  const auto& f3 = std::make_shared<Tensor<float>>(3, 224, 224);
  TensorElementMultiply(f1, f2, f3);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 6.f);
  }
}

TEST(test_tensor, mul6) {
  using namespace kuiper_infer;
  const auto& f1 = std::make_shared<Tensor<float>>(3, 224, 224);
  f1->fill(3.f);
  const auto& f2 = std::make_shared<Tensor<float>>(3, 1, 1);
  f2->fill(2.f);

  const auto& f3 = std::make_shared<Tensor<float>>(3, 224, 224);
  TensorElementMultiply(f2, f1, f3);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 6.f);
  }
}