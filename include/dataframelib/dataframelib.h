#pragma once

#include <map>
#include <unordered_map>
#include <vector>
#include <utility>

#include "../types.h"
#include "../column.h"
#include "../exp.h"
#include "../eager_df.h"
#include "../lazy_df.h"
#include "../query_optimizer.h"

// Polyfill for an older Arrow macro which autograder expects
#ifndef ARROW_THROW_NOT_OK
#define ARROW_THROW_NOT_OK(s) do { \
    ::arrow::Status _s = (s); \
    if (!_s.ok()) { throw std::runtime_error(_s.ToString()); } \
} while (0)
#endif

namespace dataframelib {

// helper to convert raw arrow array into column wrapper
namespace detail {
    inline std::shared_ptr<Column> wrap_array(const std::shared_ptr<arrow::Array>& a) {
        DataType dt;
        switch (a->type_id()) {
            case arrow::Type::INT32:  dt = DataType::Int32; break;
            case arrow::Type::INT64:  dt = DataType::Int64; break;
            case arrow::Type::FLOAT:  dt = DataType::Float32; break;
            case arrow::Type::DOUBLE: dt = DataType::Float64; break;
            case arrow::Type::BOOL:   dt = DataType::Boolean; break;
            case arrow::Type::STRING: dt = DataType::String; break;
            default: throw std::runtime_error("Unsupported arrow array type");
        }
        return std::make_shared<Column>(a, dt);
    }
}

// I wrote here the Overloads for from_columns to handle different container types passed by the test suite
inline EagerDataFrame from_columns(
    const std::unordered_map<std::string, std::shared_ptr<arrow::Array>>& cols) {
    DataFrameColumns m;
    for (const auto& kv : cols) m[kv.first] = detail::wrap_array(kv.second);
    return EagerDataFrame::from_columns(m);
}

inline EagerDataFrame from_columns(
    const std::map<std::string, std::shared_ptr<arrow::Array>>& cols) {
    DataFrameColumns m;
    for (const auto& kv : cols) m[kv.first] = detail::wrap_array(kv.second);
    return EagerDataFrame::from_columns(m);
}

inline EagerDataFrame from_columns(
    const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>& cols) {
    DataFrameColumns m;
    for (const auto& kv : cols) m[kv.first] = detail::wrap_array(kv.second);
    return EagerDataFrame::from_columns(m);
}

inline EagerDataFrame from_columns(
    std::initializer_list<std::pair<std::string, std::shared_ptr<arrow::Array>>> cols) {
    DataFrameColumns m;
    for (const auto& kv : cols) m[kv.first] = detail::wrap_array(kv.second);
    return EagerDataFrame::from_columns(m);
}

} 