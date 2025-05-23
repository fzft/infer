#include "data/tensor.h"
#include <omp.h>


namespace kuiper_infer {

    template<typename T>
    Tensor<T>::Tensor(T* raw_ptr, uint32_t size) {
        CHECK_NOTNULL(raw_ptr);
        raw_shapes_ = {size};
        data_ = arma::Cube<T>(raw_ptr, 1, size, 1, false, true);
    }

    template<typename T>
    Tensor<T>::Tensor(T* raw_ptr, uint32_t rows, uint32_t cols) {
        CHECK_NE(raw_ptr, nullptr);
        if (rows == 1) {
            raw_shapes_ = {cols};
        } else {
            raw_shapes_ = {rows, cols};
        }
        data_ = arma::Cube<T>(raw_ptr, rows, cols, 1, false, true);
    }

    template<typename T>
    Tensor<T>::Tensor(T* raw_ptr, uint32_t channels, uint32_t rows, uint32_t cols) {
        CHECK_NE(raw_ptr, nullptr);
        if (channels == 1 && rows == 1) {
            raw_shapes_ = {cols};
        } else if (channels == 1) {
            raw_shapes_ = {rows, cols};
        } else {
            raw_shapes_ = {channels, rows, cols};
        }
        data_ = arma::Cube<T>(raw_ptr, rows, cols, channels, false, true);
    }
    
    template <typename T>
    Tensor<T>::Tensor(T* raw_ptr, const std::vector<uint32_t>& shapes) {
        CHECK_NE(raw_ptr, nullptr);
        uint32_t channels = shapes.at(0);
        uint32_t rows = shapes.at(1);
        uint32_t cols = shapes.at(2);

        if (channels == 1 && rows == 1) {
            raw_shapes_ = std::vector<uint32_t>{cols};
        } else if (channels == 1) {
            raw_shapes_ = std::vector<uint32_t>{rows, cols};
        } else {
            raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
        }

        data_ = arma::Cube<T>(raw_ptr, rows, cols, channels, false, true);
    }

    template<typename T>
    Tensor<T>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
        data_ = arma::Cube<T>(rows, cols, channels);
        if (channels == 1 && rows == 1) {
            raw_shapes_ = std::vector<uint32_t>{cols};
        } else if (channels == 1) {
            raw_shapes_ = std::vector<uint32_t>{rows, cols};
        } else {
            raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
        }
    }

    template<typename T>
    Tensor<T>::Tensor(uint32_t size) {
        raw_shapes_ = {size};
        data_ = arma::Cube<T>(1, size, 1);
    }

    template<typename T>
    Tensor<T>::Tensor(uint32_t rows, uint32_t cols) {
        raw_shapes_ = {rows, cols};
        data_ = arma::Cube<T>(rows, cols, 1);
    }

    template<typename T>
    Tensor<T>::Tensor(const std::vector<uint32_t>& shapes) {
        uint32_t remaining = 3 - shapes.size();
        std::vector<uint32_t> shapes_(3, 1);
        std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

        uint32_t channels = shapes_.at(0);
        uint32_t rows = shapes_.at(1);
        uint32_t cols = shapes_.at(2);
        data_ = arma::Cube<T>(channels, rows, cols);
        if (channels == 1 && rows == 1) {
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        } else if (channels == 1) {
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        } else {
            this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
        }
    }

    template<typename T>
    uint32_t Tensor<T>::size() const {
        return this->data_.size();
    }

    template<typename T>
    uint32_t Tensor<T>::rows() const {
        return this->data_.n_rows;
    }

    template<typename T>
    uint32_t Tensor<T>::cols() const {
        return this->data_.n_cols;
    }

    template<typename T>
    uint32_t Tensor<T>::channels() const {
        return this->data_.n_slices;
    }


    template<typename T>
    void Tensor<T>::set_data(const arma::Cube<T>& data) {
        this->data_ = data;
    }

    template<typename T>
    bool Tensor<T>::empty() const {
        return this->data_.empty();
    }

    template<typename T>
    T& Tensor<T>::index(uint32_t offset) {
        return this->data_.at(offset);
    }


    template<typename T>
    const T Tensor<T>::index(uint32_t offset) const {
        return this->data_.at(offset);
    }


    template<typename T>
    const std::vector<uint32_t> Tensor<T>::shapes() const {
        return {this->channels(), this->rows(), this->cols()};
    }

    template<typename T>
    arma::Cube<T>& Tensor<T>::data() {
        return this->data_;
    }

    template<typename T>
    const arma::Cube<T>& Tensor<T>::data() const {
        return this->data_;
    }

    template<typename T>
    arma::Mat<T>& Tensor<T>::slice(uint32_t channel) {
        return this->data_.slice(channel);
    }

    template<typename T>
    const arma::Mat<T>& Tensor<T>::slice(uint32_t channel) const {
        return this->data_.slice(channel);
    }

    template<typename T>
    const T Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) const {
        return this->data_.at(row, col, channel);
    }

    template<typename T>
    T& Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) {
        return this->data_.at(row, col, channel);
    }

    template<typename T>
    void Tensor<T>::padding(const vector<uint32_t>& pads, T padding_value) {
        uint32_t pad_row1 = pads.at(0);  // up
        uint32_t pad_row2 = pads.at(1);  // bottom
        uint32_t pad_col1 = pads.at(2);  // left
        uint32_t pad_col2 = pads.at(3);  // right

        arma::Cube<T> new_data(
            this->data_.n_rows + pad_row1 + pad_row2,
            this->data_.n_cols + pad_col1 + pad_col2,
            this->data_.n_slices
        );

        new_data.fill(padding_value);
        new_data.subcube(
            pad_row1,
            pad_col1,
            0,
            new_data.n_rows - pad_row2 - 1,
            new_data.n_cols - pad_col2 - 1,
            new_data.n_slices - 1
        ) = this->data_;
        this->data_ = move(new_data);
        this->raw_shapes_ = {
            this->channels(),
            this->rows(),
            this->cols()
        };
    }

    template<typename T>
    void Tensor<T>::fill(T value) {
        this->data_.fill(value);
    }

    template<typename T>
    void Tensor<T>::fill(const std::vector<T>& values, bool row_major) {
        if (row_major) {
            const uint32_t rows = this->rows();
            const uint32_t cols = this->cols();
            const uint32_t channels = this->channels();
            const uint32_t planes = rows * cols;
            for (uint32_t c = 0; c < channels; ++c) {
                arma::Mat<T> channel_data_t(const_cast<T*>(values.data() + c * planes), rows, cols, false, true);
                this->data_.slice(c) = channel_data_t;
            }
        } else {
            std::copy(values.begin(), values.end(), this->data_.memptr());
        }
    }
    
    template<typename T>
    void Tensor<T>::ones() {
        this->fill(1);
    }

    template<typename T>
    void Tensor<T>::zeros() {
        this->fill(0);
    }

    template<>    
    void Tensor<float>::randu(float min, float max) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> dis(min, max);
        for (uint32_t i = 0; i < this->data_.size(); ++i) {
            this->index(i) = dis(gen);
        }
    }

    template<typename T>
    void Tensor<T>::randn(T mean, T std) {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<T> dis(mean, std);
        for (uint32_t i = 0; i < this->data_.size(); ++i) {
            this->index(i) = dis(gen);
        }
    }

    template<typename T>
    void Tensor<T>::transform(const function<T(const T&)> &op) {
        this->data_.transform(op);
    }

    template<typename T>
    void Tensor<T>::show() const {
        for (uint32_t i = 0; i < this->data_.n_slices; ++i) {
            LOG(INFO) << "Channel: " << i << endl;
            LOG(INFO) << this->data_.slice(i) << endl;
        }
    }

    template<typename T>
    const std::vector<uint32_t>& Tensor<T>::raw_shapes() const {
        CHECK(!this->raw_shapes_.empty());
        CHECK_LE(this->raw_shapes_.size(), 3);
        CHECK_GE(this->raw_shapes_.size(), 1);
        return this->raw_shapes_;
    }

    template<typename T>
    void Tensor<T>::reshape(const std::vector<uint32_t>& shapes, bool row_major) {
        const uint32_t origin_size = this->data_.size();
        const uint32_t current_size = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<uint32_t>());
        if (!row_major) {
            if (shapes.size() == 3) {
                this->data_ = this->data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
                this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
            } else if (shapes.size() == 2) { 
                this->data_.reshape(shapes.at(0), shapes.at(1), 1);
                this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
            } else {
                this->data_.reshape(1, shapes.at(0), 1);
                this->raw_shapes_ = {shapes.at(0)};
            }
        } else {
           if (shapes.size() == 3) {
            this->review({shapes.at(0), shapes.at(1), shapes.at(2)});
            this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
           } else if (shapes.size() == 2) {
            this->review({shapes.at(0), shapes.at(1)});
            this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
           } else {
            this->review({1, 1, shapes.at(0)});
            this->raw_shapes_ = {shapes.at(0)};
           }
        }
    }

    template<typename T>
    void Tensor<T>::flatten(bool row_major) {
        const uint32_t size = this->data_.size();
        this->reshape({size}, row_major);
    }
    
    template<typename T>
    const T* Tensor<T>::raw_ptr() const {
        return this->data_.memptr();
    }

    template<typename T>
    T* Tensor<T>::raw_ptr(uint32_t offset) {
        return this->data_.memptr() + offset;
    }

    template<typename T>
    void Tensor<T>::review(const std::vector<uint32_t>& shapes) {
        uint32_t target_channels = shapes.at(0);
        uint32_t target_rows = shapes.at(1);
        uint32_t target_cols = shapes.at(2);
        arma::Cube<T> new_data(target_channels, target_rows, target_cols);
        const uint32_t plane_size = target_rows * target_cols;
    #pragma omp parallel for
        for (uint32_t channel = 0; channel < this->data_.n_slices; ++channel) {
            const uint32_t plane_start = channel * data_.n_rows * data_.n_cols;
            for (uint32_t src_col = 0; src_col < this->data_.n_cols; ++src_col) {
            const T* col_ptr = this->data_.slice_colptr(channel, src_col);
            for (uint32_t src_row = 0; src_row < this->data_.n_rows; ++src_row) {
                const uint32_t pos_idx = plane_start + src_row * data_.n_cols + src_col;
                const uint32_t dst_ch = pos_idx / plane_size;
                const uint32_t dst_ch_offset = pos_idx % plane_size;
                const uint32_t dst_row = dst_ch_offset / target_cols;
                const uint32_t dst_col = dst_ch_offset % target_cols;
                new_data.at(dst_row, dst_col, dst_ch) = *(col_ptr + src_row);
                }
            }
        }
        this->data_ = std::move(new_data);
    }

    template class Tensor<float>;
};
