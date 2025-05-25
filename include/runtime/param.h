#pragma once
#include <string>
#include <vector>
#include "utils/status_code.h"

namespace kuiper_infer {
    struct RuntimeParameter {  /// 计算节点中的参数信息
  virtual ~RuntimeParameter() = default;

  explicit RuntimeParameter(RuntimeParameterType type = RuntimeParameterType::kParameterUnknown)
      : type(type) {}
  RuntimeParameterType type = RuntimeParameterType::kParameterUnknown;
};

struct RuntimeParameterInt : public RuntimeParameter {
  explicit RuntimeParameterInt() : RuntimeParameter(RuntimeParameterType::kParameterInt) {}

  explicit RuntimeParameterInt(int32_t param_value)
      : RuntimeParameter(RuntimeParameterType::kParameterInt), value(param_value) {}

  int32_t value = 0;
};

struct RuntimeParameterFloat : public RuntimeParameter {
  explicit RuntimeParameterFloat() : RuntimeParameter(RuntimeParameterType::kParameterFloat) {}

  explicit RuntimeParameterFloat(float param_value)
      : RuntimeParameter(RuntimeParameterType::kParameterFloat), value(param_value) {}

  float value = 0.f;
};

struct RuntimeParameterString : public RuntimeParameter {
  explicit RuntimeParameterString() : RuntimeParameter(RuntimeParameterType::kParameterString) {}

  explicit RuntimeParameterString(std::string param_value)
      : RuntimeParameter(RuntimeParameterType::kParameterString), value(std::move(param_value)) {}

  std::string value;
};

struct RuntimeParameterIntArray : public RuntimeParameter {
  explicit RuntimeParameterIntArray()
      : RuntimeParameter(RuntimeParameterType::kParameterIntArray) {}

  explicit RuntimeParameterIntArray(std::vector<int32_t> param_value)
      : RuntimeParameter(RuntimeParameterType::kParameterIntArray), value(std::move(param_value)) {}

  std::vector<int32_t> value;
};

struct RuntimeParameterFloatArray : public RuntimeParameter {
  explicit RuntimeParameterFloatArray()
      : RuntimeParameter(RuntimeParameterType::kParameterFloatArray) {}

  explicit RuntimeParameterFloatArray(std::vector<float> param_value)
      : RuntimeParameter(RuntimeParameterType::kParameterFloatArray),
        value(std::move(param_value)) {}

  std::vector<float> value;
};

struct RuntimeParameterStringArray : public RuntimeParameter {
  explicit RuntimeParameterStringArray()
      : RuntimeParameter(RuntimeParameterType::kParameterStringArray) {}

  explicit RuntimeParameterStringArray(std::vector<std::string> param_value)
      : RuntimeParameter(RuntimeParameterType::kParameterStringArray),
        value(std::move(param_value)) {}

  std::vector<std::string> value;
};

struct RuntimeParameterBool : public RuntimeParameter {
  explicit RuntimeParameterBool() : RuntimeParameter(RuntimeParameterType::kParameterBool) {}

  explicit RuntimeParameterBool(bool param_value)
      : RuntimeParameter(RuntimeParameterType::kParameterBool), value(param_value) {}

  bool value = false;
};
}

