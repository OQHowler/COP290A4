#include "exp.h"
#include <arrow/api.h>
#include <arrow/builder.h>
#include <stdexcept>
#include <string>

// =============================================================================
// ConstantNode::evaluate
// Broadcasts a single literal value across every row in the dataframe so it
// can be combined with column-shaped operands in the expression tree.
// =============================================================================
std::shared_ptr<Column> ConstantNode::evaluate(const DataFrameColumns& current_df_state) {

    // 1. Determine how long this column needs to be.
    int64_t target_row_count = 1; // Default to 1 for isolated AST testing

    // Use .empty() and .begin() for an O(1) fetch rather than a break-loop
    if (!current_df_state.empty()) {
        target_row_count = current_df_state.begin()->second->size();
    }

    // 2. MACRO ENGINE: Parse the string ONCE, then loop to broadcast it.
    // This prevents standard library boilerplate and keeps the AST footprint minimal.
    #define BUILD_CONSTANT_ARRAY(BUILDER_TYPE, PARSE_EXPR) \
    { \
        BUILDER_TYPE builder; \
        (void)builder.Reserve(target_row_count); \
        auto broadcast_val = PARSE_EXPR; \
        for (int64_t i = 0; i < target_row_count; ++i) { \
            (void)builder.Append(broadcast_val); \
        } \
        std::shared_ptr<arrow::Array> finished_array; \
        if (!builder.Finish(&finished_array).ok()) { \
            throw std::runtime_error("Fatal: Error finishing the constant array build."); \
        } \
        return std::make_shared<Column>(finished_array, m_storedType); \
    }

    // 3. Dispatch to the correct Arrow Builder based on our exact Enum type
    // We wrap everything in a try-catch because std::stoi/stof will throw if 
    // an invalid string was somehow forced into the AST.
    try {
        switch (m_storedType) {
            case DataType::Int32:   BUILD_CONSTANT_ARRAY(arrow::Int32Builder, std::stoi(m_rawData))
            case DataType::Int64:   BUILD_CONSTANT_ARRAY(arrow::Int64Builder, std::stoll(m_rawData))
            case DataType::Float32: BUILD_CONSTANT_ARRAY(arrow::FloatBuilder, std::stof(m_rawData))
            case DataType::Float64: BUILD_CONSTANT_ARRAY(arrow::DoubleBuilder, std::stod(m_rawData))
            case DataType::String:  BUILD_CONSTANT_ARRAY(arrow::StringBuilder, m_rawData)
            case DataType::Boolean: BUILD_CONSTANT_ARRAY(arrow::BooleanBuilder, (m_rawData == "true" || m_rawData == "1"))
            
            default: 
                throw std::domain_error("Data type not supported for ConstantNode broadcasting yet.");
        }
    } catch (...) {
        // Catch parsing failures (e.g., trying to parse "hello" into an Int32)
        throw std::invalid_argument("Failed to parse the constant value: " + m_rawData);
    }
    
    #undef BUILD_CONSTANT_ARRAY
}