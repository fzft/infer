#pragma once

#include <glog/logging.h>
#include "runtime/datatype.h"
#include <vector>
#include <string>

namespace kuiper_infer {
    struct RuntimeAttribute {
        RuntimeAttribute() = default;

    explicit RuntimeAttribute(std::vector<int32_t> shape, RuntimeDataType type,
                            std::vector<char> weight_data)
      : shape(std::move(shape)), type(type), weight_data(std::move(weight_data)) {}


    /**
 * @brief Attribute data
 *
 * Typically contains the binary weight values.
 */
    std::vector<char> weight_data;

    /**
 * @brief Shape of the attribute
 *
 * Describes the dimensions of the attribute data.
 */
    std::vector<int32_t> shape;

    RuntimeDataType type = RuntimeDataType::kTypeUnknown;

    template <class T>
    std::vector<T> get(bool need_clear_weight = true);
    };
    template <class T>
    std::vector<T> RuntimeAttribute::get(bool need_clear_weight) {
    CHECK(!weight_data.empty());
    CHECK(type != RuntimeDataType::kTypeUnknown);
    const uint32_t elem_size = sizeof(T);
    CHECK_EQ(weight_data.size() % elem_size, 0);
    const uint32_t weight_data_size = weight_data.size() / elem_size;

    std::vector<T> weights;
    weights.reserve(weight_data_size);
    switch (type) {
        case RuntimeDataType::kTypeFloat32: {
        static_assert(std::is_same<T, float>::value == true);
        float* weight_data_ptr = reinterpret_cast<float*>(weight_data.data());
        for (uint32_t i = 0; i < weight_data_size; ++i) {
            float weight = *(weight_data_ptr + i);
            weights.push_back(weight);
        }
        break;
        }
        default: {
        LOG(FATAL) << "Unknown weight data type: " << int32_t(type);
        }
    }
    if (need_clear_weight) {
        std::vector<char> empty_vec = std::vector<char>();
        this->weight_data.swap(empty_vec);
    }
    return weights;
    }
}