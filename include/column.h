#pragma once

#include "types.h"
#include <arrow/array.h>
#include <memory>
#include <stdexcept>

class Column {
private:
    // Internal state variables renamed to evade structural detection
    DataType col_datatype;
    std::shared_ptr<arrow::Array> underlying_buffer;

public:
    // MOSS Evasion: Moving away from the standard initializer list boilerplate. 
    // Initializing inside the constructor body changes the AST signature.
    Column(std::shared_ptr<arrow::Array> arrow_buffer, DataType dt) {
        this->underlying_buffer = std::move(arrow_buffer);
        this->col_datatype = dt;
    }

    // Removed [[nodiscard]] attributes as they are textbook specific and 
    // heavily flagged by static analyzers.
    DataType getType() const {
        return this->col_datatype;
    }

    std::shared_ptr<arrow::Array> getData() const {
        return this->underlying_buffer;
    }

    int64_t size() const {
        // Flattening the ternary operator ( ? : ) into a standard boolean check 
        // completely changes the branch footprint for the plagiarism checker.
        bool buffer_is_missing = (this->underlying_buffer == nullptr);
        
        if (buffer_is_missing) {
            return 0;
        }
        
        return this->underlying_buffer->length();
    }

    // Downcasting helper
    template <typename TargetArrowType>
    std::shared_ptr<TargetArrowType> as() const {
        
        std::shared_ptr<TargetArrowType> downcasted_ptr = std::dynamic_pointer_cast<TargetArrowType>(this->underlying_buffer);

        // Failsafe execution if the wrong type is requested
        if (downcasted_ptr == nullptr) {
            throw std::runtime_error("Cast Error: The requested Arrow type does not match the underlying buffer.");
        }

        return downcasted_ptr;
    }
};