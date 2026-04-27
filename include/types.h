#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <stdexcept>
#include <arrow/api.h>
#include <arrow/type.h>
namespace dataframelib {


// main datatypes
enum class DataType {
    Int32,
    Int64,
    Float32,
    Float64,
    String,
    Boolean
};

// Here, i wrote code for type promotion engine which handles upcasting when evaluting binary ops
inline DataType promoteTypes(DataType type_l, DataType type_r) {

    // if perfect match then no promotion
    bool types_are_identical = (type_l == type_r);
    if (types_are_identical) {
        return type_l;
    }

    // if there is floating point then mixed numeric types become floats
    bool left_has_decimals = (type_l == DataType::Float32 || type_l == DataType::Float64);
    bool right_has_decimals = (type_r == DataType::Float32 || type_r == DataType::Float64);

    if (left_has_decimals || right_has_decimals) {

        return DataType::Float64;
    }


    // if mixed integer types then I upcast to Int64
    bool is_mixed_int_combo_one = (type_l == DataType::Int32 && type_r == DataType::Int64);
    bool is_mixed_int_combo_two = (type_l == DataType::Int64 && type_r == DataType::Int32);

    if (is_mixed_int_combo_one || is_mixed_int_combo_two) {
        return DataType::Int64;
    }


    throw std::invalid_argument("Type Error: The requested data types are incompatible and cannot be promoted.");
}



inline arrow::Type::type getArrowType(DataType custom_target_type) {
    
    
    if (custom_target_type == DataType::Int32) {
        return arrow::Type::INT32;
    }
    if (custom_target_type == DataType::Int64) {
        return arrow::Type::INT64;
    }
    if (custom_target_type == DataType::Float32) {
        return arrow::Type::FLOAT;
    }
    if (custom_target_type == DataType::Float64) {
        return arrow::Type::DOUBLE;
    }
    if (custom_target_type == DataType::String) {
        return arrow::Type::STRING;
    }
    if (custom_target_type == DataType::Boolean) {
        return arrow::Type::BOOL;
    }

    throw std::invalid_argument("Mapping Error: Unrecognized internal DataType provided.");
}


inline DataType getCustomType(arrow::Type::type native_arrow_id) {
    
    
    if (native_arrow_id == arrow::Type::INT32) {
        return DataType::Int32;
    }
    if (native_arrow_id == arrow::Type::INT64) {
        return DataType::Int64;
    }
    if (native_arrow_id == arrow::Type::FLOAT) {
        return DataType::Float32;
    }
    if (native_arrow_id == arrow::Type::DOUBLE) {
        return DataType::Float64;
    }
    if (native_arrow_id == arrow::Type::STRING) {
        return DataType::String;
    }
    if (native_arrow_id == arrow::Type::BOOL) {
        return DataType::Boolean;
    }

    throw std::invalid_argument("Mapping Error: This Arrow type is not supported by the DataFrameLib engine.");
}

}