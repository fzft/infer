#pragma once

#include <memory>
#include <string>
#include <vector>
#include "data/tensor.h"
#include "datatype.h"

namespace kuiper_infer{
    template <typename T>
    struct RuntimeOperandBase {
        explicit RuntimeOperandBase() = default;

        explicit RuntimeOperandBase(std::string name, std::vector<int32_t> shapes,
                              std::vector<std::shared_ptr<Tensor<T>>> datas, RuntimeDataType type)
      : name(std::move(name)), shapes(std::move(shapes)), datas(std::move(datas)), type(type) {}

        explicit RuntimeOperandBase(std::string name, std::vector<int32_t> shapes, uint32_t data_size,
                                    RuntimeDataType type)
            : name(std::move(name)), shapes(std::move(shapes)), type(type) {
            datas.resize(data_size);
        }

        size_t size() const;

        /// Name of the operand
        std::string name;

        /// Shape of the operand
        std::vector<int32_t> shapes;

        /// Vector containing operand data
        std::vector<std::shared_ptr<Tensor<T>>> datas;

        /// Data type of the operand
        RuntimeDataType type = RuntimeDataType::kTypeUnknown;
    };

    template <typename T>
    size_t RuntimeOperandBase<T>::size() const {
    if (shapes.empty()) {
        return 0;
    }
    size_t size = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
    return size;
    }

    using RuntimeOperand = RuntimeOperandBase<float>;
};