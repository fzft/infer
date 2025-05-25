#include <glog/logging.h>
#include <iostream>
#include <gtest/gtest.h>
#include "runtime/ir.h"
#include "runtime/op.h"

using namespace std;


TEST(test_runtime, runtime_graph_input_init1) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<RuntimeOperator>> operators;
  uint32_t op_size = 3;
  for (uint32_t i = 0; i < op_size; ++i) {
    const auto& runtime_operator = std::make_shared<RuntimeOperator>();
    const auto& runtime_operand1 = std::make_shared<RuntimeOperand>();
    runtime_operand1->shapes = {3, 32, 32};
    runtime_operand1->type = RuntimeDataType::kTypeFloat32;

    const auto& runtime_operand2 = std::make_shared<RuntimeOperand>();
    runtime_operand2->shapes = {3, 64, 64};
    runtime_operand2->type = RuntimeDataType::kTypeFloat32;

    runtime_operator->input_operands.insert({std::string("size1"), runtime_operand1});
    runtime_operator->input_operands.insert({std::string("size2"), runtime_operand2});

    operators.push_back(runtime_operator);
  }
  ASSERT_EQ(operators.size(), 3);
  RuntimeOperatorUtils<float>::InitOperatorInput(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto& op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    const uint32_t size1 = op->input_operands["size1"]->datas.size();
    const uint32_t size2 = op->input_operands["size2"]->datas.size();
    ASSERT_EQ(size1, 3);
    ASSERT_EQ(size1, size2);
  }

  RuntimeOperatorUtils<float>::InitOperatorInput(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto& op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    const uint32_t size1 = op->input_operands["size1"]->datas.size();
    const uint32_t size2 = op->input_operands["size2"]->datas.size();
    ASSERT_EQ(size1, 3);
    ASSERT_EQ(size1, size2);
  }
}