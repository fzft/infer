#pragma once

#include "data/tensor.h"

using namespace std;

namespace kuiper_infer {

/*
* @brief 广播两个张量
* @param tensor1 张量1
* @param tensor2 张量2
* @return 广播后的张量
*/
template <typename T>
std::tuple<std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> TensorBroadcast(
    const std::shared_ptr<Tensor<T>>& tensor1, const std::shared_ptr<Tensor<T>>& tensor2);


/*
* @brief 两个张量逐元素相乘
* @param tensor1 张量1
* @param tensor2 张量2
* @param output_tensor 输出张量
*/
template <typename T>
void TensorElementMultiply(const std::shared_ptr<Tensor<T>>& tensor1,
                           const std::shared_ptr<Tensor<T>>& tensor2,
                           const std::shared_ptr<Tensor<T>>& output_tensor);

/*
* @brief 判断两个张量是否相同
* @param a 张量a
* @param b 张量b
* @param threshold 阈值
* @return 是否相同
*/
template <typename T>
bool TensorIsSame(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b,
                  T threshold = 1e-5f);


/*
* @brief 两个张量逐元素相加
* @param tensor1 张量1
* @param tensor2 张量2
* @return 相加后的张量
*/
template <typename T>
std::shared_ptr<Tensor<T>> TensorElementAdd(const std::shared_ptr<Tensor<T>>& tensor1,
                                            const std::shared_ptr<Tensor<T>>& tensor2);

template <typename T>
void TensorElementAdd(const std::shared_ptr<Tensor<T>>& tensor1,
                      const std::shared_ptr<Tensor<T>>& tensor2,
                      const std::shared_ptr<Tensor<T>>& output_tensor);   

/*
* @brief 两个张量逐元素相乘
* @param tensor1 张量1
* @param tensor2 张量2
* @return 相乘后的张量
*/
template <typename T>
std::shared_ptr<Tensor<T>> TensorElementMultiply(const std::shared_ptr<Tensor<T>>& tensor1,
                                                 const std::shared_ptr<Tensor<T>>& tensor2);    

/*
* @brief 创建一个张量
* @param channels 通道数
* @param rows 行数
* @param cols 列数
* @return 创建的张量
*/
template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t channels, uint32_t rows, uint32_t cols);    


/*
* @brief 创建一个张量
* @param rows 行数
* @param cols 列数
* @return 创建的张量
*/
template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t rows, uint32_t cols);

/*
* @brief 创建一个张量
* @param size 大小
* @return 创建的张量
*/
template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t size);

/*
* @brief 创建一个张量
* @param shapes 形状
* @return 创建的张量
*/
template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(const std::vector<uint32_t>& shapes);

/*
* @brief 克隆一个张量
* @param tensor 张量
* @return 克隆后的张量
*/
template <typename T>
std::shared_ptr<Tensor<T>> TensorClone(std::shared_ptr<Tensor<T>> tensor);


template <typename T>
tuple<shared_ptr<Tensor<T>>, shared_ptr<Tensor<T>>> TensorBroadcast(
    const shared_ptr<Tensor<T>>& tensor1, const shared_ptr<Tensor<T>>& tensor2) {
    CHECK(tensor1 != nullptr && tensor2 != nullptr);
    if (tensor1->shapes() == tensor2->shapes()) {
        return {tensor1, tensor2};
    } else {
        CHECK(tensor1->channels() == tensor2->channels());
        if (tensor2->rows() == 1 && tensor2->cols() == 1) {
        std::shared_ptr<Tensor<T>> new_tensor =
            TensorCreate<T>(tensor2->channels(), tensor1->rows(), tensor1->cols());
        CHECK(tensor2->size() == tensor2->channels());
        for (uint32_t c = 0; c < tensor2->channels(); ++c) {
            T* new_tensor_ptr = new_tensor->matrix_raw_ptr(c);
            std::fill(new_tensor_ptr, new_tensor_ptr + new_tensor->plane_size(), tensor2->index(c));
        }
        return {tensor1, new_tensor};
        } else if (tensor1->rows() == 1 && tensor1->cols() == 1) {
        std::shared_ptr<Tensor<T>> new_tensor =
            TensorCreate<T>(tensor1->channels(), tensor2->rows(), tensor2->cols());
        CHECK(tensor1->size() == tensor1->channels());
            for (uint32_t c = 0; c < tensor1->channels(); ++c) {
            T* new_tensor_ptr = new_tensor->matrix_raw_ptr(c);
            std::fill(new_tensor_ptr, new_tensor_ptr + new_tensor->plane_size(), tensor1->index(c));
        }
        return {new_tensor, tensor2};
        } else {
        LOG(FATAL) << "Broadcast shape is not adapting!";
        return {tensor1, tensor2};
        }
    }
}

template <typename T>
bool TensorIsSame(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b,
                    T threshold) {
    CHECK(a != nullptr && b != nullptr);
    if (a->shape() != b->shape()) {
        return false;
    }
    bool is_same = arma::approx_equal(a->data(), b->data(), "absdiff", threshold);
    return is_same;
}

template <typename T>
shared_ptr<Tensor<T>> TensorElementAdd(const shared_ptr<Tensor<T>>& tensor1,
                                        const shared_ptr<Tensor<T>>& tensor2) {
    CHECK(tensor1 != nullptr && tensor2 != nullptr);
    if (tensor1->shapes() == tensor2->shapes()) {
        std::shared_ptr<Tensor<T>> output_tensor = TensorCreate<T>(tensor1->shapes());
        output_tensor->set_data(tensor1->data() + tensor2->data());
        return output_tensor;
    } else {
        // broadcast
        CHECK(tensor1->channels() == tensor2->channels()) << "Tensors shape are not adapting";
        const auto& [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
        CHECK(input_tensor1->shapes() == input_tensor2->shapes());
        std::shared_ptr<Tensor<T>> output_tensor = TensorCreate<T>(input_tensor1->shapes());
        output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
        return output_tensor;
    }
}


template <typename T>
void TensorElementAdd(const shared_ptr<Tensor<T>>& tensor1,
                        const shared_ptr<Tensor<T>>& tensor2,
                        const shared_ptr<Tensor<T>>& output_tensor) {
        CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
    if (tensor1->shapes() == tensor2->shapes()) {
        CHECK(tensor1->shapes() == output_tensor->shapes());
        output_tensor->set_data(tensor1->data() + tensor2->data());
    } else {
        CHECK(tensor1->channels() == tensor2->channels()) << "Tensors shape are not adapting";
        const auto& [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
        CHECK(output_tensor->shapes() == input_tensor1->shapes() &&
            output_tensor->shapes() == input_tensor2->shapes());
        output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
    }
}

template <typename T>
shared_ptr<Tensor<T>> TensorElementMultiply(const shared_ptr<Tensor<T>>& tensor1,
                                            const shared_ptr<Tensor<T>>& tensor2) {
    CHECK(tensor1 != nullptr && tensor2 != nullptr);
    if (tensor1->shapes() == tensor2->shapes()) {
        std::shared_ptr<Tensor<T>> output_tensor = TensorCreate<T>(tensor1->shapes());
        output_tensor->set_data(tensor1->data() % tensor2->data());
        return output_tensor;
    } else {
        // broadcast
        CHECK(tensor1->channels() == tensor2->channels()) << "Tensors shape are not adapting";
        const auto& [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
        CHECK(input_tensor1->shapes() == input_tensor2->shapes());
        std::shared_ptr<Tensor<T>> output_tensor = TensorCreate<T>(input_tensor1->shapes());
        output_tensor->set_data(input_tensor1->data() % input_tensor2->data());
        return output_tensor;
    }
}

template <typename T>
void TensorElementMultiply(const shared_ptr<Tensor<T>>& tensor1,
                            const shared_ptr<Tensor<T>>& tensor2,
                            const shared_ptr<Tensor<T>>& output_tensor) {
    CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
    if (tensor1->shapes() == tensor2->shapes()) {
        CHECK(tensor1->shapes() == output_tensor->shapes());
        output_tensor->set_data(tensor1->data() % tensor2->data());
    } else {
        CHECK(tensor1->channels() == tensor2->channels()) << "Tensors shape are not adapting";
        const auto& [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
        CHECK(output_tensor->shapes() == input_tensor1->shapes() &&
            output_tensor->shapes() == input_tensor2->shapes());
        output_tensor->set_data(input_tensor1->data() % input_tensor2->data());
    }
    
}

template <typename T>
shared_ptr<Tensor<T>> TensorCreate(uint32_t channels, uint32_t rows, uint32_t cols) {
    return make_shared<Tensor<T>>(channels, rows, cols);
}

template <typename T>
shared_ptr<Tensor<T>> TensorCreate(uint32_t rows, uint32_t cols) {
    return make_shared<Tensor<T>>(1, rows, cols);
}

template <typename T>
shared_ptr<Tensor<T>> TensorCreate(uint32_t size) {
    return make_shared<Tensor<T>>(1, 1, size);
}

template <typename T>
shared_ptr<Tensor<T>> TensorCreate(const vector<uint32_t>& shapes) {
    CHECK(!shapes.empty() && shapes.size() <= 3);
    if (shapes.size() == 1) {
        return TensorCreate<T>(shapes.front());
    } else if (shapes.size() == 2) {
        return TensorCreate<T>(shapes.front(), shapes.back());
    } else {
        return TensorCreate<T>(shapes.front(), shapes[1], shapes.back());
    }
}

template <typename T>
std::shared_ptr<Tensor<T>> TensorClone(std::shared_ptr<Tensor<T>> tensor) {
    CHECK(tensor != nullptr);
    return make_shared<Tensor<T>>(*tensor);
}

} // namespace kuiper_infer