#include "exp.h"

#include <arrow/api.h>
#include <arrow/builder.h>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "lazy_df.h"
#include "query_optimizer.h"
#include <fstream>
#include <sstream>

namespace dataframelib {
class ArrayDataFetcher {
private:
    DataType internal_col_type;
    std::shared_ptr<arrow::Array> generic_array_ptr;
    
    // Cached concrete pointers
    arrow::Int32Array* ptr_int32 = nullptr;
    arrow::Int64Array* ptr_int64 = nullptr;
    arrow::FloatArray* ptr_float32 = nullptr;
    arrow::DoubleArray* ptr_float64 = nullptr;
    arrow::BooleanArray* ptr_boolean = nullptr;
    arrow::StringArray* ptr_string = nullptr;

public:
    ArrayDataFetcher(std::shared_ptr<Column> target_column) {
        
        internal_col_type = target_column->getType();
        generic_array_ptr = target_column->getData();
        
        if (internal_col_type == DataType::Int32) {
            ptr_int32 = static_cast<arrow::Int32Array*>(generic_array_ptr.get());
            return;
        }
        if (internal_col_type == DataType::Int64) {
            ptr_int64 = static_cast<arrow::Int64Array*>(generic_array_ptr.get());
            return;
        }
        if (internal_col_type == DataType::Float32) {
            ptr_float32 = static_cast<arrow::FloatArray*>(generic_array_ptr.get());
            return;
        }
        if (internal_col_type == DataType::Float64) {
            ptr_float64 = static_cast<arrow::DoubleArray*>(generic_array_ptr.get());
            return;
        }
        if (internal_col_type == DataType::Boolean) {
            ptr_boolean = static_cast<arrow::BooleanArray*>(generic_array_ptr.get());
            return;
        }
        if (internal_col_type == DataType::String) {
            ptr_string = static_cast<arrow::StringArray*>(generic_array_ptr.get());
            return;
        }
    }

    DataType fetch_datatype() const { 
        return internal_col_type; 
    }

    bool has_valid_data(int64_t row_index) const { 
        return generic_array_ptr->IsValid(row_index); 
    }

    // upcasting all numeric types to double to do universal math 
    double extract_as_double(int64_t row_index) const {
        if (ptr_int32 != nullptr) return static_cast<double>(ptr_int32->Value(row_index));
        if (ptr_int64 != nullptr) return static_cast<double>(ptr_int64->Value(row_index));
        if (ptr_float32 != nullptr) return static_cast<double>(ptr_float32->Value(row_index));
        if (ptr_float64 != nullptr) return ptr_float64->Value(row_index);
        return 0.0; 
    }

    int64_t extract_as_integer(int64_t row_index) const {
        if (ptr_int32 != nullptr) return static_cast<int64_t>(ptr_int32->Value(row_index));
        if (ptr_int64 != nullptr) return ptr_int64->Value(row_index);
        return 0;
    }
    
    std::string extract_string(int64_t row_index) const {
        if (ptr_string != nullptr) return ptr_string->GetString(row_index);
        return "";
    }
    
    bool extract_boolean(int64_t row_index) const {
        if (ptr_boolean != nullptr) return ptr_boolean->Value(row_index);
        return false;
    }
};

using BufferAccessor = ArrayDataFetcher;


// Helper template to  construct broadcast arrays
template <typename ArrowBuilder, typename CppNativeType>
std::shared_ptr<Column> construct_broadcast_array(int64_t row_count, CppNativeType broadcast_val, DataType target_type) {
    ArrowBuilder array_maker;
    (void)array_maker.Reserve(row_count);
    
    for (int64_t idx = 0; idx < row_count; ++idx) {
        (void)array_maker.Append(broadcast_val);
    }
    
    std::shared_ptr<arrow::Array> finished_buffer;
    if (!array_maker.Finish(&finished_buffer).ok()) {
        throw std::runtime_error("Fatal: Arrow failed to construct broadcast array.");
    }
    
    return std::make_shared<Column>(finished_buffer, target_type);
}


std::shared_ptr<Column> ConstantNode::evaluate(const DataFrameColumns& current_table_state) {
    
    int64_t needed_rows = 1; 
    bool has_existing_data = !current_table_state.empty();
    if (has_existing_data) {
        needed_rows = current_table_state.begin()->second->size();
    }

    try {
        if (internal_datatype == DataType::Int32) {
            return construct_broadcast_array<arrow::Int32Builder, int32_t>(needed_rows, std::stoi(stringified_value), internal_datatype);
        } 
        else if (internal_datatype == DataType::Int64) {
            return construct_broadcast_array<arrow::Int64Builder, int64_t>(needed_rows, std::stoll(stringified_value), internal_datatype);
        } 
        else if (internal_datatype == DataType::Float32) {
            return construct_broadcast_array<arrow::FloatBuilder, float>(needed_rows, std::stof(stringified_value), internal_datatype);
        } 
        else if (internal_datatype == DataType::Float64) {
            return construct_broadcast_array<arrow::DoubleBuilder, double>(needed_rows, std::stod(stringified_value), internal_datatype);
        } 
        else if (internal_datatype == DataType::String) {
            return construct_broadcast_array<arrow::StringBuilder, std::string>(needed_rows, stringified_value, internal_datatype);
        } 
        else if (internal_datatype == DataType::Boolean) {
            bool bool_val = (stringified_value == "true" || stringified_value == "1");
            return construct_broadcast_array<arrow::BooleanBuilder, bool>(needed_rows, bool_val, internal_datatype);
        } 
        else {
            throw std::domain_error("AST Error: Unrecognized constant type.");
        }
    } catch (...) {
        throw std::invalid_argument("AST Error: Failed to parse literal string: " + stringified_value);
    }
}
std::shared_ptr<Column> BinaryOp::evaluate(const DataFrameColumns& execution_environment) {
    
    // Bottom-up AST execution
    std::shared_ptr<Column> evaluated_left = lhs_node->evaluate(execution_environment);
    std::shared_ptr<Column> evaluated_right = rhs_node->evaluate(execution_environment);

    int64_t left_row_count = evaluated_left->size();
    bool row_mismatch = (left_row_count != evaluated_right->size());
    
    if (row_mismatch) {
        throw std::invalid_argument("Fatal AST Error: Cannot evaluate binary operation on columns of different lengths.");
    }

    ArrayDataFetcher left_accessor(evaluated_left);
    ArrayDataFetcher right_accessor(evaluated_right);

    bool op_is_add = (operation_token == "+");
    bool op_is_sub = (operation_token == "-");
    bool op_is_mul = (operation_token == "*");
    bool op_is_div = (operation_token == "/");
    bool op_is_mod = (operation_token == "%");
    bool operation_is_math = (op_is_add || op_is_sub || op_is_mul || op_is_div || op_is_mod);
    
    bool operation_is_comparison = (operation_token == "==" || operation_token == "!=" || 
                                    operation_token == "<"  || operation_token == "<=" || 
                                    operation_token == ">"  || operation_token == ">=");
                                    
    bool operation_is_logical = (operation_token == "&" || operation_token == "|");

    // matchematical ops
    if (operation_is_math) {
        
        DataType l_type = left_accessor.fetch_datatype();
        DataType r_type = right_accessor.fetch_datatype();
        
        bool involves_strings = (l_type == DataType::String || r_type == DataType::String);
        
        if (involves_strings) {
            if (l_type != DataType::String || r_type != DataType::String) {
                throw std::invalid_argument("Type Error: Cannot add a numeric and a non-numeric column.");
            }
            if (!op_is_add) {
                throw std::invalid_argument("Type Error: Only addition (+) is supported for strings.");
            }
            
            arrow::StringBuilder concat_builder;
            (void)concat_builder.Reserve(left_row_count);
            
            for(int64_t row_idx = 0; row_idx < left_row_count; ++row_idx) {
                bool l_valid = left_accessor.has_valid_data(row_idx);
                bool r_valid = right_accessor.has_valid_data(row_idx);
                
                if (!l_valid || !r_valid) { 
                    (void)concat_builder.AppendNull(); 
                } else { 
                    std::string combined_text = left_accessor.extract_string(row_idx) + right_accessor.extract_string(row_idx);
                    (void)concat_builder.Append(combined_text); 
                }
            }
            
            std::shared_ptr<arrow::Array> final_strings; 
            (void)concat_builder.Finish(&final_strings);
            return std::make_shared<Column>(final_strings, DataType::String);
        }

        DataType target_promotion_type = promoteTypes(l_type, r_type);
        bool promote_to_floating_point = (target_promotion_type == DataType::Float64 || target_promotion_type == DataType::Float32);

        if (promote_to_floating_point) {
            arrow::DoubleBuilder dbl_builder;
            (void)dbl_builder.Reserve(left_row_count);
            
            for(int64_t row_idx = 0; row_idx < left_row_count; ++row_idx) {
                if (!left_accessor.has_valid_data(row_idx) || !right_accessor.has_valid_data(row_idx)) { 
                    (void)dbl_builder.AppendNull(); 
                    continue; 
                }
                
                double val_l = left_accessor.extract_as_double(row_idx);
                double val_r = right_accessor.extract_as_double(row_idx);
                
                if (op_is_add) { (void)dbl_builder.Append(val_l + val_r); }
                else if (op_is_sub) { (void)dbl_builder.Append(val_l - val_r); }
                else if (op_is_mul) { (void)dbl_builder.Append(val_l * val_r); }
                else if (op_is_div) {
                    if (val_r == 0.0) throw std::runtime_error("Execution Error: Floating point division by zero.");
                    (void)dbl_builder.Append(val_l / val_r);
                }
                else {
                    throw std::runtime_error("Execution Error: Cannot perform modulo operator on floating point types.");
                }
            }
            
            std::shared_ptr<arrow::Array> final_floats; 
            (void)dbl_builder.Finish(&final_floats);
            return std::make_shared<Column>(final_floats, DataType::Float64); 
            
        } else {
           
            if (target_promotion_type == DataType::Int32) {
                arrow::Int32Builder int_builder;
                (void)int_builder.Reserve(left_row_count);
                
                for(int64_t row_idx = 0; row_idx < left_row_count; ++row_idx) {
                    if (!left_accessor.has_valid_data(row_idx) || !right_accessor.has_valid_data(row_idx)) { 
                        (void)int_builder.AppendNull(); 
                        continue; 
                    }
                    
                    int32_t val_l = static_cast<int32_t>(left_accessor.extract_as_integer(row_idx));
                    int32_t val_r = static_cast<int32_t>(right_accessor.extract_as_integer(row_idx));
                    
                    if (op_is_add) { (void)int_builder.Append(val_l + val_r); }
                    else if (op_is_sub) { (void)int_builder.Append(val_l - val_r); }
                    else if (op_is_mul) { (void)int_builder.Append(val_l * val_r); }
                    else if (op_is_div) {
                        if (val_r == 0) throw std::runtime_error("Execution Error: Integer division by zero.");
                        (void)int_builder.Append(val_l / val_r);
                    }
                    else if (op_is_mod) {
                        if (val_r == 0) throw std::runtime_error("Execution Error: Integer modulo by zero.");
                        (void)int_builder.Append(val_l % val_r);
                    }
                }
                
                std::shared_ptr<arrow::Array> final_ints; 
                (void)int_builder.Finish(&final_ints);
                return std::make_shared<Column>(final_ints, DataType::Int32);
                
            } else {
                arrow::Int64Builder int_builder;
                (void)int_builder.Reserve(left_row_count);
                
                for(int64_t row_idx = 0; row_idx < left_row_count; ++row_idx) {
                    if (!left_accessor.has_valid_data(row_idx) || !right_accessor.has_valid_data(row_idx)) { 
                        (void)int_builder.AppendNull(); 
                        continue; 
                    }
                    
                    int64_t val_l = left_accessor.extract_as_integer(row_idx);
                    int64_t val_r = right_accessor.extract_as_integer(row_idx);
                    
                    if (op_is_add) { (void)int_builder.Append(val_l + val_r); }
                    else if (op_is_sub) { (void)int_builder.Append(val_l - val_r); }
                    else if (op_is_mul) { (void)int_builder.Append(val_l * val_r); }
                    else if (op_is_div) {
                        if (val_r == 0) throw std::runtime_error("Execution Error: Integer division by zero.");
                        (void)int_builder.Append(val_l / val_r);
                    }
                    else if (op_is_mod) {
                        if (val_r == 0) throw std::runtime_error("Execution Error: Integer modulo by zero.");
                        (void)int_builder.Append(val_l % val_r);
                    }
                }
                
                std::shared_ptr<arrow::Array> final_ints; 
                (void)int_builder.Finish(&final_ints);
                return std::make_shared<Column>(final_ints, DataType::Int64);
            }
        }
    }

    // comparision ops
    else if (operation_is_comparison) {
        arrow::BooleanBuilder comparison_builder;
        (void)comparison_builder.Reserve(left_row_count);
        

bool left_is_str = (left_accessor.fetch_datatype() == DataType::String);
bool right_is_str = (right_accessor.fetch_datatype() == DataType::String);

if (left_is_str != right_is_str) {
    throw std::invalid_argument("Type Error: Cannot compare a string with a numeric column.");
}
bool check_as_strings = (left_is_str && right_is_str);
        
        for(int64_t row_idx = 0; row_idx < left_row_count; ++row_idx) {
            if (!left_accessor.has_valid_data(row_idx) || !right_accessor.has_valid_data(row_idx)) { 
                (void)comparison_builder.AppendNull(); 
                continue; 
            }
            
            bool truth_val = false;
            
            if (check_as_strings) {
                std::string str_l = left_accessor.extract_string(row_idx); 
                std::string str_r = right_accessor.extract_string(row_idx);
                
                if (operation_token == "==") { truth_val = (str_l == str_r); }
                else if (operation_token == "!=") { truth_val = (str_l != str_r); }
                else if (operation_token == "<")  { truth_val = (str_l < str_r); }
                else if (operation_token == "<=") { truth_val = (str_l <= str_r); }
                else if (operation_token == ">")  { truth_val = (str_l > str_r); }
                else if (operation_token == ">=") { truth_val = (str_l >= str_r); }
            } else {
                double num_l = left_accessor.extract_as_double(row_idx); 
                double num_r = right_accessor.extract_as_double(row_idx);
                
                if (operation_token == "==") { truth_val = (num_l == num_r); }
                else if (operation_token == "!=") { truth_val = (num_l != num_r); }
                else if (operation_token == "<")  { truth_val = (num_l < num_r); }
                else if (operation_token == "<=") { truth_val = (num_l <= num_r); }
                else if (operation_token == ">")  { truth_val = (num_l > num_r); }
                else if (operation_token == ">=") { truth_val = (num_l >= num_r); }
            }
            (void)comparison_builder.Append(truth_val);
        }
        
        std::shared_ptr<arrow::Array> final_bools; 
        (void)comparison_builder.Finish(&final_bools);
        return std::make_shared<Column>(final_bools, DataType::Boolean);
    }
    // logical ops
    else if (operation_is_logical) {
        bool operands_are_boolean = (left_accessor.fetch_datatype() == DataType::Boolean && right_accessor.fetch_datatype() == DataType::Boolean);
        
        if (!operands_are_boolean) {
            throw std::invalid_argument("Type Error: Bitwise/Logical operators require boolean columns.");
        }
        
        arrow::BooleanBuilder logical_builder;
        (void)logical_builder.Reserve(left_row_count);
        
        for(int64_t row_idx = 0; row_idx < left_row_count; ++row_idx) {
            if (!left_accessor.has_valid_data(row_idx) || !right_accessor.has_valid_data(row_idx)) { 
                (void)logical_builder.AppendNull(); 
                continue; 
            }
            
            bool val_l = left_accessor.extract_boolean(row_idx); 
            bool val_r = right_accessor.extract_boolean(row_idx);
            
            if (operation_token == "&") { (void)logical_builder.Append(val_l && val_r); }
            else if (operation_token == "|") { (void)logical_builder.Append(val_l || val_r); }
        }
        
        std::shared_ptr<arrow::Array> final_logical_bools; 
        (void)logical_builder.Finish(&final_logical_bools);
        return std::make_shared<Column>(final_logical_bools, DataType::Boolean);
    }

    throw std::runtime_error("AST Parsing Error: Could not resolve binary operator '" + operation_token + "'.");
}

// unary ops
std::shared_ptr<Column> UnaryOp::evaluate(const DataFrameColumns& execution_context) {
    
    std::shared_ptr<Column> downstream_data = child_operand->evaluate(execution_context);
    int64_t total_rows = downstream_data->size();
    BufferAccessor downstream_buffer(downstream_data);

    if (action_token == "is_null" || action_token == "is_not_null") {
        
        arrow::BooleanBuilder null_check_builder; 
        (void)null_check_builder.Reserve(total_rows);
        
        for (int64_t row = 0; row < total_rows; ++row) {
            bool data_exists = downstream_buffer.has_valid_data(row);
            bool check_result = (action_token == "is_null") ? !data_exists : data_exists;
            (void)null_check_builder.Append(check_result);
        }
        
        std::shared_ptr<arrow::Array> final_checks; 
        (void)null_check_builder.Finish(&final_checks);
        return std::make_shared<Column>(final_checks, DataType::Boolean);
    }
    
    else if (action_token == "~") {
        
        if (downstream_buffer.fetch_datatype() != DataType::Boolean) {
            throw std::runtime_error("AST Error: The NOT operator (~) requires a Boolean column.");
        }
        
        arrow::BooleanBuilder inversion_builder; 
        (void)inversion_builder.Reserve(total_rows);
        
        for (int64_t row = 0; row < total_rows; ++row) {
            if (!downstream_buffer.has_valid_data(row)) { 
                (void)inversion_builder.AppendNull(); 
            } else { 
                bool current_val = downstream_buffer.extract_boolean(row);
                (void)inversion_builder.Append(!current_val); 
            }
        }
        
        std::shared_ptr<arrow::Array> final_inversions; 
        (void)inversion_builder.Finish(&final_inversions);
        return std::make_shared<Column>(final_inversions, DataType::Boolean);
    }
    
    else if (action_token == "abs") {
        
        DataType target_type = downstream_buffer.fetch_datatype();
        
        if (target_type == DataType::Int32) {
            arrow::Int32Builder int_builder; 
            (void)int_builder.Reserve(total_rows);
            for (int64_t row = 0; row < total_rows; ++row) {
                if (!downstream_buffer.has_valid_data(row)) { 
                    (void)int_builder.AppendNull(); 
                } else { 
                    (void)int_builder.Append(std::abs(static_cast<int32_t>(downstream_buffer.extract_as_integer(row)))); 
                }
            }
            std::shared_ptr<arrow::Array> computed_abs; 
            (void)int_builder.Finish(&computed_abs);
            return std::make_shared<Column>(computed_abs, DataType::Int32);
            
        } 
        else if (target_type == DataType::Int64) {
            arrow::Int64Builder int_builder; 
            (void)int_builder.Reserve(total_rows);
            for (int64_t row = 0; row < total_rows; ++row) {
                if (!downstream_buffer.has_valid_data(row)) { 
                    (void)int_builder.AppendNull(); 
                } else { 
                    (void)int_builder.Append(std::abs(downstream_buffer.extract_as_integer(row))); 
                }
            }
            std::shared_ptr<arrow::Array> computed_abs; 
            (void)int_builder.Finish(&computed_abs);
            return std::make_shared<Column>(computed_abs, DataType::Int64);
            
        } 
else if (target_type == DataType::Float32 || target_type == DataType::Float64) {
            arrow::DoubleBuilder float_builder; 
            (void)float_builder.Reserve(total_rows);
            for (int64_t row = 0; row < total_rows; ++row) {
                if (!downstream_buffer.has_valid_data(row)) { 
                    (void)float_builder.AppendNull(); 
                } else { 
                    (void)float_builder.Append(std::abs(downstream_buffer.extract_as_double(row))); 
                }
            }
            std::shared_ptr<arrow::Array> computed_abs; 
            (void)float_builder.Finish(&computed_abs);
            // FIX: Force tag to Float64 because we built a DoubleArray
            return std::make_shared<Column>(computed_abs, DataType::Float64);
        }
        
        throw std::runtime_error("AST Error: abs() requires a numeric column.");
    }
    
    else if (action_token == "length") {
        
        if (downstream_buffer.fetch_datatype() != DataType::String) {
            throw std::runtime_error("AST Error: length() requires a String column.");
        }
        
        arrow::Int32Builder length_builder; 
        (void)length_builder.Reserve(total_rows);
        
        for (int64_t row = 0; row < total_rows; ++row) {
            if (!downstream_buffer.has_valid_data(row)) { 
                (void)length_builder.AppendNull(); 
            } else { 
                std::string cell_text = downstream_buffer.extract_string(row);
                (void)length_builder.Append(static_cast<int32_t>(cell_text.length())); 
            }
        }
        
        std::shared_ptr<arrow::Array> final_lengths; 
        (void)length_builder.Finish(&final_lengths);
        return std::make_shared<Column>(final_lengths, DataType::Int32);
    }
    
    else if (action_token == "to_lower" || action_token == "to_upper") {
        
        if (downstream_buffer.fetch_datatype() != DataType::String) {
            throw std::runtime_error("AST Error: Case conversion requires a String column.");
        }
        
        arrow::StringBuilder string_builder; 
        (void)string_builder.Reserve(total_rows);
        
        for (int64_t row = 0; row < total_rows; ++row) {
            if (!downstream_buffer.has_valid_data(row)) { 
                (void)string_builder.AppendNull(); 
                continue; 
            }
            
            std::string text_val = downstream_buffer.extract_string(row);
            
            if (action_token == "to_lower") {
                std::transform(text_val.begin(), text_val.end(), text_val.begin(), ::tolower);
            } else {
                std::transform(text_val.begin(), text_val.end(), text_val.begin(), ::toupper);
            }
            
            (void)string_builder.Append(text_val);
        }
        
        std::shared_ptr<arrow::Array> final_strings; 
        (void)string_builder.Finish(&final_strings);
        return std::make_shared<Column>(final_strings, DataType::String);
    }
    
    throw std::runtime_error("AST Error: Unsupported unary operator token -> " + action_token);
}

// method ops
std::shared_ptr<Column> MethodOp::evaluate(const DataFrameColumns& execution_context) {
    
    std::shared_ptr<Column> primary_data = base_target->evaluate(execution_context);
    std::shared_ptr<Column> search_args = argument_node->evaluate(execution_context); 

    int64_t total_rows = primary_data->size();
    
    BufferAccessor primary_buffer(primary_data);
    BufferAccessor args_buffer(search_args);

    if (primary_buffer.fetch_datatype() != DataType::String || args_buffer.fetch_datatype() != DataType::String) {
        throw std::invalid_argument("AST Error: String methods strictly require String operands.");
    }

    arrow::BooleanBuilder bool_builder; 
    (void)bool_builder.Reserve(total_rows);

    for (int64_t row = 0; row < total_rows; ++row) {
        
        bool both_valid = primary_buffer.has_valid_data(row) && args_buffer.has_valid_data(row);
        if (!both_valid) { 
            (void)bool_builder.AppendNull(); 
            continue; 
        }
        
        std::string base_text = primary_buffer.extract_string(row);
        std::string query_text = args_buffer.extract_string(row);

        bool evaluation_result = false;
        
        if (function_id == "contains") {
            auto match_idx = base_text.find(query_text);
            evaluation_result = (match_idx < base_text.length());
        } 
        else if (function_id == "starts_with") {
            auto prefix_idx = base_text.rfind(query_text, 0);
            evaluation_result = (prefix_idx == 0);
        } 
        else if (function_id == "ends_with") {
            bool is_query_smaller = (query_text.length() <= base_text.length());
            if (is_query_smaller) {
                size_t start_pos = base_text.length() - query_text.length();
                evaluation_result = (base_text.compare(start_pos, query_text.length(), query_text) == 0);
            }
        }
        
        (void)bool_builder.Append(evaluation_result);
    }
    
    std::shared_ptr<arrow::Array> final_bools; 
    (void)bool_builder.Finish(&final_bools);
    return std::make_shared<Column>(final_bools, DataType::Boolean);
}


std::shared_ptr<Column> AliasNode::evaluate(const DataFrameColumns& execution_context) {
    // The structural mapping is tracked by the df.
    //and AST evaluates the underlying node and bubbles the physical data upward.
    std::shared_ptr<Column> materialized_data = original_expr->evaluate(execution_context);
    return materialized_data;
}



    /**
 * @brief Triggers the execution of the lazy computation DAG.
 * @note This method automatically invokes the QueryOptimizer before materializing the data.
 * @return An EagerDataFrame containing the fully materialized results.
 */
EagerDataFrame LazyDataFrame::collect() {
    QueryOptimizer opt;
    LazyDataFrame optimized_plan = opt.optimize(*this);
    return optimized_plan.current_head_node->evaluate();
}


    /**
 * @brief Generates a Graphviz DOT visualization of the optimized computation DAG.
 * @param image_output_path The file path (e.g., "plan.png") where the render will be saved.
 */
 
void LazyDataFrame::explain(const std::string& image_output_path) const {
    QueryOptimizer opt;
    LazyDataFrame optimized_plan = opt.optimize(*this);
    
    std::stringstream graph_buffer;
    graph_buffer << "digraph ComputationPlan {\n";
    graph_buffer << "  node [shape=box, style=filled, fontname=\"Helvetica\"];\n";
    
    optimized_plan.generate_dot_markup(optimized_plan.current_head_node, graph_buffer);
    
    graph_buffer << "}\n";

    std::string interim_text_file = image_output_path + ".dot";
    std::ofstream disk_handle(interim_text_file);
    disk_handle << graph_buffer.str();
    disk_handle.close();

    std::string shell_instruction = "dot -Tpng " + interim_text_file + " -o " + image_output_path;
    int shell_exit_status = std::system(shell_instruction.c_str());
    
    if (shell_exit_status != 0) {
        std::cerr << "Rendering Warning: Graphviz failed.\n";
    }
}


}