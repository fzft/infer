#pragma once
#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <numeric>
#include <vector>

using namespace std;

namespace kuiper_infer {
    template<typename T>
    class Tensor {
        public:
            explicit Tensor() = default;
            explicit Tensor(T* raw_ptr, uint32_t size);
            explicit Tensor(T* raw_ptr, uint32_t rows, uint32_t cols);
            explicit Tensor(T* raw_ptr, uint32_t channels, uint32_t rows, uint32_t cols);
            explicit Tensor(T* raw_ptr, const vector<uint32_t>& shapes);


            explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
            explicit Tensor(uint32_t size);
            explicit Tensor(uint32_t rows, uint32_t cols);
            explicit Tensor(const vector<uint32_t>& shapes);


            // ----- function -----

            /*
            * @brief 获取张量的大小
            * @return 张量的大小
            */
            uint32_t size() const;

            /*
            * @brief 获取张量的行数
            */
            uint32_t rows() const;

            /*
            * @brief 获取张量的列数
            */  
            uint32_t cols() const;

            /*
            * @brief 获取张量的通道数
            */
            uint32_t channels() const;

            
            void set_data(const arma::Cube<T>& data);


            bool empty() const;


            /*
            * @brief 获取张量指定位置的值
            * @param offset 偏移量
            * @return 指定位置的值的引用
            */
            T& index(uint32_t offset);

            /*
            * @brief 获取张量指定位置的值
            * @param offset 偏移量
            * @return 指定位置的值
            */
            const T index(uint32_t offset) const;


            /*
            * @brief 获取张量的形状
            * @return 张量的形状
            */
            const vector<uint32_t> shapes() const;


            const std::vector<uint32_t>& raw_shapes() const;

            /*
            * @brief 获取张量的数据
            * @return 张量的数据
            */
            arma::Cube<T>& data();

            const arma::Cube<T>& data() const;

            arma::Mat<T>& slice(uint32_t channel);

            const arma::Mat<T>& slice(uint32_t channel) const;

           /*
            * @brief 获取张量指定位置的值
            * @param channel 通道
            * @param row 行
            * @param col 列
            * @return 指定位置的值
            */
            const T at(uint32_t channel, uint32_t row, uint32_t col) const;

           /*
            * @brief 获取张量指定位置的值
            * @param channel 通道
            * @param row 行
            * @param col 列
            * @return 指定位置的值的引用
            */
            T& at(uint32_t channel, uint32_t row, uint32_t col);

           /*
            * @brief 填充张量
            * @param pads 填充的尺寸
            * @param padding_value 填充的值
            */
            void padding(const vector<uint32_t>& pads, T padding_value);

           /*
            * @brief 填充张量
            * @param value 填充的值
            */
            void fill(T value);

           
           /*
            * @brief 填充张量
            * @param values 填充的值
            * @param row_major 是否按行填充
            */
            void fill(const vector<T>& values, bool row_major = true);

           /*
            * @brief 填充张量
            * @param value 填充的值
            */
            void ones();

           /*
            * @brief 填充张量
            * @param value 填充的值
            */


            void zeros();

           /*
            * @brief 填充张量
            * @param min 最小值
            * @param max 最大值
            */
            void randu(T min, T max);

           /*
            * @brief 填充张量
            * @param mean 均值
            * @param std 标准差
            */
            void randn(T mean, T std);


            void transform(const function<T(const T&)> &op);

            void review(const vector<uint32_t>& shapes);
           
           /*
            * @brief 显示张量
            */
            void show() const;


            void reshape(const vector<uint32_t>& shape, bool row_major = false);


            void flatten(bool row_major = true);

            const T* raw_ptr() const;


            T* raw_ptr(uint32_t offset);

        private:
            arma::Cube<T> data_;
            vector<uint32_t> raw_shapes_;
    };
};