#pragma once

#include "types.h"
#include <arrow/array.h>
#include <memory>
#include <stdexcept>

namespace dataframelib {

class Column {
private:
    // logical datatype tag
    DataType col_datatype;
    // physical arrow array
    std::shared_ptr<arrow::Array> underlying_buffer;

public:
    Column(std::shared_ptr<arrow::Array> arrow_buffer, DataType dt) {
        this->underlying_buffer = std::move(arrow_buffer);
        this->col_datatype = dt;
    }

    DataType getType() const {
        return this->col_datatype;
    }

    std::shared_ptr<arrow::Array> getData() const {
        return this->underlying_buffer;
    }

    int64_t size() const {
        // Safety check for uninitialized buffer
        bool buffer_is_missing = (this->underlying_buffer == nullptr);
        
        if (buffer_is_missing) {
            return 0;
        }
        
        return this->underlying_buffer->length();
    }
    // helper to downcasr generic arrow array to a specific concrete type
    template <typename TargetArrowType>
    std::shared_ptr<TargetArrowType> as() const {
        
        std::shared_ptr<TargetArrowType> downcasted_ptr = std::dynamic_pointer_cast<TargetArrowType>(this->underlying_buffer);

        // trow error if cast to wrong type
        if (downcasted_ptr == nullptr) {
            throw std::runtime_error("Cast Error: The requested Arrow type does not match the underlying buffer.");
        }

        return downcasted_ptr;
    }
};

}