#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <optional>

#include "column.h"
#include "exp.h"

#include <arrow/io/api.h>
#include <arrow/csv/api.h>
#include <arrow/table.h>
#include <arrow/compute/api.h>

#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

namespace dataframelib {

    
class GroupedDataFrame; // Forward declaration


// these are various Helper functions which i wrote later to slice, mask, sort, and join Arrow arrays.
namespace engine_utils {

    template <typename ArrowT>
    std::shared_ptr<Column> execute_head(std::shared_ptr<Column> source_col, int64_t threshold, DataType type_tag) {
        auto underlying_arr = source_col->as<ArrowT>();
        
        // this Prevents out-of-bounds slicing
        int64_t valid_limit = std::min<int64_t>(threshold, underlying_arr->length());
        std::shared_ptr<arrow::Array> sliced_data = underlying_arr->Slice(0, valid_limit);
        
        return std::make_shared<Column>(sliced_data, type_tag);
    }

template <typename ArrowT>
    void compute_sort_permutation(std::shared_ptr<Column> source_col,
                                  std::vector<int64_t>& index_map,
                                  bool is_asc) {
        auto raw_buffer = source_col->as<ArrowT>();
        std::stable_sort(index_map.begin(), index_map.end(),
            [&raw_buffer, is_asc](int64_t l, int64_t r) {
                bool lv = raw_buffer->IsValid(l);
                bool rv = raw_buffer->IsValid(r);
                if (!lv && !rv) return false;
                if (!lv) return false;     
                if (!rv) return true;
                auto a = raw_buffer->Value(l);
                auto b = raw_buffer->Value(r);
                return is_asc ? (a < b) : (a > b);
            });
    }

    template <typename ArrowT, typename BuilderT>
    std::shared_ptr<Column> gather_by_indices(std::shared_ptr<Column> source_col, const std::vector<int64_t>& permutation, DataType type_tag) {
        auto raw_buffer = source_col->as<ArrowT>();
        BuilderT arr_maker;
        (void)arr_maker.Reserve(permutation.size());
        
        for (int64_t p_idx : permutation) {
            if (raw_buffer->IsValid(p_idx)) {
                (void)arr_maker.Append(raw_buffer->Value(p_idx));
            } else {
                (void)arr_maker.AppendNull();
            }
        }
        
        std::shared_ptr<arrow::Array> finished_buffer;
        (void)arr_maker.Finish(&finished_buffer);
        return std::make_shared<Column>(finished_buffer, type_tag);
    }

    template <typename ArrowT, typename BuilderT>
    std::shared_ptr<Column> apply_boolean_mask(std::shared_ptr<Column> source_col, std::shared_ptr<arrow::BooleanArray> logic_mask, DataType type_tag) {
        auto raw_buffer = source_col->as<ArrowT>();
        BuilderT arr_maker;
        
        int64_t total_elements = logic_mask->length();
        for (int64_t row = 0; row < total_elements; ++row) {
            
            bool should_keep = logic_mask->IsValid(row) && logic_mask->Value(row);
            if (!should_keep) continue;
            
            if (raw_buffer->IsValid(row)) {
                (void)arr_maker.Append(raw_buffer->Value(row));
            } else {
                (void)arr_maker.AppendNull();
            }
        }
        
        std::shared_ptr<arrow::Array> finished_buffer;
        (void)arr_maker.Finish(&finished_buffer);
        return std::make_shared<Column>(finished_buffer, type_tag);
    }

    template <typename ArrowT, typename NativeT>
    void map_join_relationships(std::shared_ptr<Column> col_l, std::shared_ptr<Column> col_r, const std::string& join_mode, std::vector<int64_t>& out_l_idx, std::vector<int64_t>& out_r_idx) {
        
        auto buf_l = col_l->as<ArrowT>();
        auto buf_r = col_r->as<ArrowT>();
        
        int64_t right_len = buf_r->length();
        std::unordered_map<NativeT, std::vector<int64_t>> hash_table;
        hash_table.reserve(right_len);
        
        for (int64_t r = 0; r < right_len; ++r) {
            if (buf_r->IsValid(r)) {
                NativeT actual_key = NativeT(buf_r->Value(r));
                hash_table[actual_key].push_back(r);
            }
        }

        int64_t left_len = buf_l->length();
        bool is_full_outer = (join_mode == "outer");
        std::vector<char> right_matched_flags(right_len, 0);

        for (int64_t l = 0; l < left_len; ++l) {
            if (!buf_l->IsValid(l)) {
                if (join_mode == "left" || is_full_outer) { 
                    out_l_idx.push_back(l); 
                    out_r_idx.push_back(-1); 
                }
                continue;
            }
            
            NativeT search_key = NativeT(buf_l->Value(l));
            auto found_it = hash_table.find(search_key);
            
            if (found_it != hash_table.end()) {
                for (int64_t match : found_it->second) {
                    out_l_idx.push_back(l);
                    out_r_idx.push_back(match);
                    
                    if (join_mode == "right" || is_full_outer) {
                        right_matched_flags[match] = 1;
                    }
                }
            } else if (join_mode == "left" || is_full_outer) {
                out_l_idx.push_back(l);
                out_r_idx.push_back(-1);
            }
        }

        if (join_mode == "right" || is_full_outer) {
            for (int64_t r = 0; r < right_len; ++r) {
                if (right_matched_flags[r] == 0) { 
                    out_l_idx.push_back(-1); 
                    out_r_idx.push_back(r); 
                }
            }
        }
    }

    template <typename ArrowT, typename BuilderT>
    std::shared_ptr<Column> build_joined_column(std::shared_ptr<Column> source_col, const std::vector<int64_t>& index_blueprint, DataType type_tag) {
        auto raw_buffer = source_col->as<ArrowT>();
        BuilderT arr_maker;
        (void)arr_maker.Reserve(index_blueprint.size());
        
        for (int64_t blueprint_idx : index_blueprint) {
            if (blueprint_idx == -1 || !raw_buffer->IsValid(blueprint_idx)) {
                (void)arr_maker.AppendNull();
            } else {
                (void)arr_maker.Append(raw_buffer->Value(blueprint_idx));
            }
        }
        
        std::shared_ptr<arrow::Array> finished_buffer;
        (void)arr_maker.Finish(&finished_buffer);
        return std::make_shared<Column>(finished_buffer, type_tag);
    }
template <typename ArrowT, typename BuilderT>
    std::shared_ptr<Column> build_coalesced_key_column(
        std::shared_ptr<Column> left_col, std::shared_ptr<Column> right_col,
        const std::vector<int64_t>& l_blueprint, const std::vector<int64_t>& r_blueprint,
        DataType type_tag) {
        
        auto l_raw = left_col->as<ArrowT>();
        auto r_raw = right_col->as<ArrowT>();
        BuilderT arr_maker;
        (void)arr_maker.Reserve(l_blueprint.size());
        
        for (size_t i = 0; i < l_blueprint.size(); ++i) {
            int64_t l_idx = l_blueprint[i];
            int64_t r_idx = r_blueprint[i];
            
            if (l_idx != -1 && l_raw->IsValid(l_idx)) {
                (void)arr_maker.Append(l_raw->Value(l_idx));
            } else if (r_idx != -1 && r_raw->IsValid(r_idx)) {
                (void)arr_maker.Append(r_raw->Value(r_idx));
            } else {
                (void)arr_maker.AppendNull();
            }
        }
        
        std::shared_ptr<arrow::Array> finished_buffer;
        (void)arr_maker.Finish(&finished_buffer);
        return std::make_shared<Column>(finished_buffer, type_tag);
    }

}


class EagerDataFrame {
private:
    DataFrameColumns internal_storage;
    std::vector<std::string> insertion_order;

public:
    EagerDataFrame() = default;

    public:
    int64_t num_rows() const {
        if (internal_storage.empty()) return 0;
        return internal_storage.begin()->second->size();
    }
    
    int64_t num_columns() const {
        return internal_storage.size();
    }

    static EagerDataFrame from_columns(const DataFrameColumns& input_map) {
        EagerDataFrame constructed_df;
        for (const auto& kv_pair : input_map) {
            constructed_df.insert_mock_column(kv_pair.first, kv_pair.second);
        }
        return constructed_df;
    }

    GroupedDataFrame group_by(const std::vector<std::string>& keys) const;

    void insert_mock_column(std::string header, std::shared_ptr<Column> col_data) {
        internal_storage[header] = col_data;
        
        auto it = std::find(insertion_order.begin(), insertion_order.end(), header);
        if (it == insertion_order.end()) {
            insertion_order.push_back(header);
        }
    }

    // i have returned a new df here to maintaini mmutability
    EagerDataFrame with_column(std::string target_name, std::shared_ptr<Exp> math_tree) const {
        
        EagerDataFrame new_df = *this; 
        
        // here, the expression is evaluate against the current data
        std::shared_ptr<Column> generated_col = math_tree->evaluate(this->internal_storage);
    
        // the column is tracked here for I/O
        auto it = std::find(new_df.insertion_order.begin(), new_df.insertion_order.end(), target_name);
        if (it == new_df.insertion_order.end()) {
            new_df.insertion_order.push_back(target_name);
        }
        
        new_df.internal_storage[target_name] = generated_col;
        return new_df; 
    }

    std::shared_ptr<Column> fetch_column(std::string target_name) const {
        if (internal_storage.find(target_name) == internal_storage.end()) {
            throw std::invalid_argument("DataFrame Error: Column not found -> " + target_name);
        }
        return internal_storage.at(target_name);
    }


    /**
 * @brief Selects specific columns from the DataFrame.
 * @param requested_cols A vector of column names to retain.
 * @return A new EagerDataFrame containing only the requested columns.
 */
    EagerDataFrame select(const std::vector<std::string>& requested_cols) const {
        EagerDataFrame selection_df;
        for (const auto& header : requested_cols) {
            selection_df.insert_mock_column(header, fetch_column(header));
        }
        return selection_df;
    }

    EagerDataFrame head(int64_t row_limit) const {
        EagerDataFrame sliced_df;
        
        for (const auto& col_id : insertion_order) {
            std::shared_ptr<Column> current_col = internal_storage.at(col_id);
            DataType tag = current_col->getType();
            std::shared_ptr<Column> processed_col;

            if (tag == DataType::Int32) processed_col = engine_utils::execute_head<arrow::Int32Array>(current_col, row_limit, tag);
            else if (tag == DataType::Int64) processed_col = engine_utils::execute_head<arrow::Int64Array>(current_col, row_limit, tag);
            else if (tag == DataType::Float32) processed_col = engine_utils::execute_head<arrow::FloatArray>(current_col, row_limit, tag);
            else if (tag == DataType::Float64) processed_col = engine_utils::execute_head<arrow::DoubleArray>(current_col, row_limit, tag);
            else if (tag == DataType::Boolean) processed_col = engine_utils::execute_head<arrow::BooleanArray>(current_col, row_limit, tag);
            else if (tag == DataType::String) processed_col = engine_utils::execute_head<arrow::StringArray>(current_col, row_limit, tag);
            else throw std::runtime_error("Head Error: Datatype not supported.");
            
            sliced_df.insert_mock_column(col_id, processed_col);
        }
        return sliced_df;
    }
EagerDataFrame sort(const std::vector<std::string>& keys, bool asc) const {
        if (keys.empty() || insertion_order.empty()) return *this;

        int64_t n = num_rows();
        std::vector<int64_t> permutation(n);
        for (int64_t r = 0; r < n; ++r) permutation[r] = r;

        const auto& store = internal_storage;
// NEW
std::stable_sort(permutation.begin(), permutation.end(),
    [&store, &keys, asc](int64_t l, int64_t r) {
        for (const auto& k : keys) {
            auto col = store.at(k);
            DataType dt = col->getType();
            std::optional<bool> decision;
            
            auto compare_with = [&](auto arr) {
                bool lv = arr->IsValid(l), rv = arr->IsValid(r);
                if (!lv && !rv) { decision = std::nullopt; return; }
                if (!lv) { decision = false; return; } // Nulls always go LAST
                if (!rv) { decision = true; return; }  // Valid always goes BEFORE Null
                
                auto a = arr->Value(l), b = arr->Value(r);
                if (a == b) { decision = std::nullopt; return; }
                decision = asc ? (a < b) : (a > b);
            };

            if      (dt == DataType::Int32)   compare_with(col->as<arrow::Int32Array>());
            else if (dt == DataType::Int64)   compare_with(col->as<arrow::Int64Array>());
            else if (dt == DataType::Float32) compare_with(col->as<arrow::FloatArray>());
            else if (dt == DataType::Float64) compare_with(col->as<arrow::DoubleArray>());
            else if (dt == DataType::String)  compare_with(col->as<arrow::StringArray>());
            else if (dt == DataType::Boolean) compare_with(col->as<arrow::BooleanArray>());
            
            if (decision.has_value()) return decision.value();
        }
        return false;
    });

        EagerDataFrame ordered_df;
        for (const auto& col_id : insertion_order) {
            auto current = internal_storage.at(col_id);
            DataType rdt = current->getType();
            std::shared_ptr<Column> rebuilt;
            if      (rdt == DataType::Int32)   rebuilt = engine_utils::gather_by_indices<arrow::Int32Array, arrow::Int32Builder>(current, permutation, rdt);
            else if (rdt == DataType::Int64)   rebuilt = engine_utils::gather_by_indices<arrow::Int64Array, arrow::Int64Builder>(current, permutation, rdt);
            else if (rdt == DataType::Float32) rebuilt = engine_utils::gather_by_indices<arrow::FloatArray, arrow::FloatBuilder>(current, permutation, rdt);
            else if (rdt == DataType::Float64) rebuilt = engine_utils::gather_by_indices<arrow::DoubleArray, arrow::DoubleBuilder>(current, permutation, rdt);
            else if (rdt == DataType::String)  rebuilt = engine_utils::gather_by_indices<arrow::StringArray, arrow::StringBuilder>(current, permutation, rdt);
            else if (rdt == DataType::Boolean) rebuilt = engine_utils::gather_by_indices<arrow::BooleanArray, arrow::BooleanBuilder>(current, permutation, rdt);
            ordered_df.insert_mock_column(col_id, rebuilt);
        }
        return ordered_df;
    }


    /**
 * @brief Filters rows based on a boolean expression AST.
 * @param ast_tree A boolean expression tree used to evaluate each row.
 * @return A new EagerDataFrame containing only rows where the expression evaluates to true.
 */

    EagerDataFrame filter(std::shared_ptr<Exp> ast_tree) {
        std::shared_ptr<Column> mask_col = ast_tree->evaluate(internal_storage);
        
        if (mask_col->getType() != DataType::Boolean) {
            throw std::invalid_argument("Filter Error: The expression did not evaluate to a Boolean column.");
        }

        auto extracted_mask = mask_col->as<arrow::BooleanArray>();
        
        // code for chec king if mask is false
        int64_t true_count = 0;
        for (int64_t i = 0; i < extracted_mask->length(); ++i) {
            if (extracted_mask->IsValid(i) && extracted_mask->Value(i)) true_count++;
        }

        // If the filter removes all rows, return an empty schema using head(0)
        if (true_count == 0) {
            return this->head(0); 
        }

        EagerDataFrame masked_df;

        for (const auto& col_id : insertion_order) {
            std::shared_ptr<Column> current_col = internal_storage.at(col_id);
            DataType dt = current_col->getType();
            std::shared_ptr<Column> filtered_col;

            if (dt == DataType::Int32) filtered_col = engine_utils::apply_boolean_mask<arrow::Int32Array, arrow::Int32Builder>(current_col, extracted_mask, dt);
            else if (dt == DataType::Int64) filtered_col = engine_utils::apply_boolean_mask<arrow::Int64Array, arrow::Int64Builder>(current_col, extracted_mask, dt);
            else if (dt == DataType::Float32) filtered_col = engine_utils::apply_boolean_mask<arrow::FloatArray, arrow::FloatBuilder>(current_col, extracted_mask, dt);
            else if (dt == DataType::Float64) filtered_col = engine_utils::apply_boolean_mask<arrow::DoubleArray, arrow::DoubleBuilder>(current_col, extracted_mask, dt);
            else if (dt == DataType::Boolean) filtered_col = engine_utils::apply_boolean_mask<arrow::BooleanArray, arrow::BooleanBuilder>(current_col, extracted_mask, dt);
            else if (dt == DataType::String) filtered_col = engine_utils::apply_boolean_mask<arrow::StringArray, arrow::StringBuilder>(current_col, extracted_mask, dt);
            else throw std::runtime_error("Filter Error: Datatype not supported.");
            
            masked_df.insert_mock_column(col_id, filtered_col);
        }
        return masked_df;
    }


    /**
 * @brief Performs a relational join with another DataFrame.
 * @param target_table The right DataFrame to join with.
 * @param keys A vector of column names to join on (currently supports single-key).
 * @param method The join algorithm ("inner", "left", "right", "outer").
 * @return A new EagerDataFrame containing the joined data.
 */
 
    EagerDataFrame join(const EagerDataFrame& target_table, const std::vector<std::string>& keys, const std::string& method) const {
        if (keys.size() != 1) throw std::invalid_argument("Join Error: Only single-column joins are implemented.");
        
        const std::string& primary_key = keys.front();
        auto left_join_col = internal_storage.at(primary_key);
        auto right_join_col = target_table.internal_storage.at(primary_key);

        if (left_join_col->getType() != right_join_col->getType()) {
            throw std::invalid_argument("Join Error: Primary key data types do not match.");
        }

        std::vector<int64_t> l_blueprint, r_blueprint;
        DataType join_dt = left_join_col->getType();

        if (join_dt == DataType::Int32) engine_utils::map_join_relationships<arrow::Int32Array, int32_t>(left_join_col, right_join_col, method, l_blueprint, r_blueprint);
        else if (join_dt == DataType::Int64) engine_utils::map_join_relationships<arrow::Int64Array, int64_t>(left_join_col, right_join_col, method, l_blueprint, r_blueprint);
        else if (join_dt == DataType::Float32) engine_utils::map_join_relationships<arrow::FloatArray, float>(left_join_col, right_join_col, method, l_blueprint, r_blueprint);
        else if (join_dt == DataType::Float64) engine_utils::map_join_relationships<arrow::DoubleArray, double>(left_join_col, right_join_col, method, l_blueprint, r_blueprint);
        else if (join_dt == DataType::Boolean) engine_utils::map_join_relationships<arrow::BooleanArray, bool>(left_join_col, right_join_col, method, l_blueprint, r_blueprint);
        else if (join_dt == DataType::String) engine_utils::map_join_relationships<arrow::StringArray, std::string>(left_join_col, right_join_col, method, l_blueprint, r_blueprint);
        else throw std::invalid_argument("Join Error: Unsupported key datatype.");

        EagerDataFrame final_joined_table;
 
        // Constructing Left Table Columns
        for (const auto& l_header : insertion_order) {
            auto src_col = internal_storage.at(l_header);
            DataType dt = src_col->getType();
            std::shared_ptr<Column> materialized_col;
            
            // Coalesce primary keys for outer/right joins to avoid null gaps
            if (l_header == primary_key && (method == "outer" || method == "right" || method == "full")) {
                auto r_src_col = target_table.internal_storage.at(primary_key);
                if (dt == DataType::Int32) materialized_col = engine_utils::build_coalesced_key_column<arrow::Int32Array, arrow::Int32Builder>(src_col, r_src_col, l_blueprint, r_blueprint, dt);
                else if (dt == DataType::Int64) materialized_col = engine_utils::build_coalesced_key_column<arrow::Int64Array, arrow::Int64Builder>(src_col, r_src_col, l_blueprint, r_blueprint, dt);
                else if (dt == DataType::Float32) materialized_col = engine_utils::build_coalesced_key_column<arrow::FloatArray, arrow::FloatBuilder>(src_col, r_src_col, l_blueprint, r_blueprint, dt);
                else if (dt == DataType::Float64) materialized_col = engine_utils::build_coalesced_key_column<arrow::DoubleArray, arrow::DoubleBuilder>(src_col, r_src_col, l_blueprint, r_blueprint, dt);
                else if (dt == DataType::Boolean) materialized_col = engine_utils::build_coalesced_key_column<arrow::BooleanArray, arrow::BooleanBuilder>(src_col, r_src_col, l_blueprint, r_blueprint, dt);
                else if (dt == DataType::String) materialized_col = engine_utils::build_coalesced_key_column<arrow::StringArray, arrow::StringBuilder>(src_col, r_src_col, l_blueprint, r_blueprint, dt);
            } else {
                if (dt == DataType::Int32) materialized_col = engine_utils::build_joined_column<arrow::Int32Array, arrow::Int32Builder>(src_col, l_blueprint, dt);
                else if (dt == DataType::Int64) materialized_col = engine_utils::build_joined_column<arrow::Int64Array, arrow::Int64Builder>(src_col, l_blueprint, dt);
                else if (dt == DataType::Float32) materialized_col = engine_utils::build_joined_column<arrow::FloatArray, arrow::FloatBuilder>(src_col, l_blueprint, dt);
                else if (dt == DataType::Float64) materialized_col = engine_utils::build_joined_column<arrow::DoubleArray, arrow::DoubleBuilder>(src_col, l_blueprint, dt);
                else if (dt == DataType::Boolean) materialized_col = engine_utils::build_joined_column<arrow::BooleanArray, arrow::BooleanBuilder>(src_col, l_blueprint, dt);
                else if (dt == DataType::String) materialized_col = engine_utils::build_joined_column<arrow::StringArray, arrow::StringBuilder>(src_col, l_blueprint, dt);
            }
            
            final_joined_table.insert_mock_column(l_header, materialized_col);
        }

        // Constructing Right Table Columns
        for (const auto& r_header : target_table.insertion_order) {
            if (r_header == primary_key) continue;
            
            auto src_col = target_table.internal_storage.at(r_header);
            DataType dt = src_col->getType();
            std::shared_ptr<Column> materialized_col;
            
            if (dt == DataType::Int32) materialized_col = engine_utils::build_joined_column<arrow::Int32Array, arrow::Int32Builder>(src_col, r_blueprint, dt);
            else if (dt == DataType::Int64) materialized_col = engine_utils::build_joined_column<arrow::Int64Array, arrow::Int64Builder>(src_col, r_blueprint, dt);
            else if (dt == DataType::Float32) materialized_col = engine_utils::build_joined_column<arrow::FloatArray, arrow::FloatBuilder>(src_col, r_blueprint, dt);
            else if (dt == DataType::Float64) materialized_col = engine_utils::build_joined_column<arrow::DoubleArray, arrow::DoubleBuilder>(src_col, r_blueprint, dt);
            else if (dt == DataType::Boolean) materialized_col = engine_utils::build_joined_column<arrow::BooleanArray, arrow::BooleanBuilder>(src_col, r_blueprint, dt);
            else if (dt == DataType::String) materialized_col = engine_utils::build_joined_column<arrow::StringArray, arrow::StringBuilder>(src_col, r_blueprint, dt);
            
            std::string alias_name = r_header;
            if (std::find(final_joined_table.insertion_order.begin(), final_joined_table.insertion_order.end(), r_header) != final_joined_table.insertion_order.end()) {
                alias_name += "_right";
            }
            final_joined_table.insert_mock_column(alias_name, materialized_col);
        }
        return final_joined_table;
    }

   // disk input and output
    void write_csv(const std::string& destination_path) const {
        std::vector<std::shared_ptr<arrow::Field>> arrow_fields;
        std::vector<std::shared_ptr<arrow::Array>> arrow_data;

        for (const auto& header : insertion_order) {
            auto wrapped_col = internal_storage.at(header);
            DataType dt = wrapped_col->getType();

            if (dt == DataType::Int32) {
                arrow_fields.push_back(arrow::field(header, arrow::int32()));
                arrow_data.push_back(wrapped_col->as<arrow::Int32Array>());
            } else if (dt == DataType::Int64) {
                arrow_fields.push_back(arrow::field(header, arrow::int64()));
                arrow_data.push_back(wrapped_col->as<arrow::Int64Array>());
            } else if (dt == DataType::Float32) {
                arrow_fields.push_back(arrow::field(header, arrow::float32()));
                arrow_data.push_back(wrapped_col->as<arrow::FloatArray>());
            } else if (dt == DataType::Float64) {
                arrow_fields.push_back(arrow::field(header, arrow::float64()));
                arrow_data.push_back(wrapped_col->as<arrow::DoubleArray>());
            } else if (dt == DataType::Boolean) {
                arrow_fields.push_back(arrow::field(header, arrow::boolean()));
                arrow_data.push_back(wrapped_col->as<arrow::BooleanArray>());
            } else if (dt == DataType::String) {
                arrow_fields.push_back(arrow::field(header, arrow::utf8()));
                arrow_data.push_back(wrapped_col->as<arrow::StringArray>());
            } else {
                throw std::runtime_error("Export Error: Type not supported by CSV writer.");
            }
        }

        auto schema = std::make_shared<arrow::Schema>(arrow_fields);
        auto export_table = arrow::Table::Make(schema, arrow_data);

        auto file_stream = arrow::io::FileOutputStream::Open(destination_path);
        if (!file_stream.ok()) throw std::runtime_error("I/O Error: Cannot lock file for CSV writing.");
        
        auto write_status = arrow::csv::WriteCSV(*export_table, arrow::csv::WriteOptions::Defaults(), file_stream.ValueOrDie().get());
        if (!write_status.ok()) throw std::runtime_error("I/O Error: CSV engine failed.");
    }

    void write_parquet(const std::string& destination_path) const {
        std::vector<std::shared_ptr<arrow::Field>> arrow_fields;
        std::vector<std::shared_ptr<arrow::Array>> arrow_data;
        
        for (const auto& header : insertion_order) {
            auto wrapped_col = internal_storage.at(header);
            DataType dt = wrapped_col->getType();

            if (dt == DataType::Int32) {
                arrow_fields.push_back(arrow::field(header, arrow::int32()));
                arrow_data.push_back(wrapped_col->as<arrow::Int32Array>());
            } else if (dt == DataType::Int64) {
                arrow_fields.push_back(arrow::field(header, arrow::int64()));
                arrow_data.push_back(wrapped_col->as<arrow::Int64Array>());
            } else if (dt == DataType::Float32) {
                arrow_fields.push_back(arrow::field(header, arrow::float32()));
                arrow_data.push_back(wrapped_col->as<arrow::FloatArray>());
            } else if (dt == DataType::Float64) {
                arrow_fields.push_back(arrow::field(header, arrow::float64()));
                arrow_data.push_back(wrapped_col->as<arrow::DoubleArray>());
            } else if (dt == DataType::Boolean) {
                arrow_fields.push_back(arrow::field(header, arrow::boolean()));
                arrow_data.push_back(wrapped_col->as<arrow::BooleanArray>());
            } else if (dt == DataType::String) {
                arrow_fields.push_back(arrow::field(header, arrow::utf8()));
                arrow_data.push_back(wrapped_col->as<arrow::StringArray>());
            } else {
                throw std::runtime_error("Export Error: Type not supported by Parquet writer.");
            }
        }
        
        auto schema = std::make_shared<arrow::Schema>(arrow_fields);
        auto export_table = arrow::Table::Make(schema, arrow_data);

        auto file_stream = arrow::io::FileOutputStream::Open(destination_path);
        if (!file_stream.ok()) throw std::runtime_error("I/O Error: Cannot lock file for Parquet writing.");
        
        auto write_status = parquet::arrow::WriteTable(*export_table, arrow::default_memory_pool(), file_stream.ValueOrDie(), 1024 * 1024);
        if (!write_status.ok()) throw std::runtime_error("I/O Error: Parquet engine failed.");
    }
};


// helpers for AGGREGATION TEMPLATE 

namespace aggregation_utils {

    template <typename ArrowT, typename NativeT>
    void map_hash_groups(std::shared_ptr<Column> key_column, std::vector<std::vector<int64_t>>& out_groups, std::vector<int64_t>& out_first_seen) {
        auto raw_buffer = key_column->as<ArrowT>();
        std::unordered_map<NativeT, size_t> hash_locator;
        
        for (int64_t row = 0; row < raw_buffer->length(); ++row) {
            if (!raw_buffer->IsValid(row)) continue;
            
            NativeT search_key = NativeT(raw_buffer->Value(row));
            auto found = hash_locator.find(search_key);
            
            if (found == hash_locator.end()) {
                hash_locator[search_key] = out_groups.size();
                out_first_seen.push_back(row);
                out_groups.push_back({row});
            } else {
                out_groups[found->second].push_back(row);
            }
        }
    }

    template <typename ArrowT, typename BuilderT, typename NativeT>
    std::shared_ptr<Column> execute_numeric_agg(std::shared_ptr<Column> source_col, const std::vector<std::vector<int64_t>>& mapped_groups, const std::string& operation, DataType type_tag) {
        auto raw_buffer = source_col->as<ArrowT>();
        BuilderT arr_maker;
        (void)arr_maker.Reserve(mapped_groups.size());

        for (const auto& group : mapped_groups) {
            if (operation == "count") {
                int32_t valid_elements = 0;
                for (int64_t row : group) { if (raw_buffer->IsValid(row)) valid_elements++; }
                (void)arr_maker.Append(valid_elements);
            } else if (operation == "sum") {
                NativeT running_total = 0; bool has_data = false;
                for (int64_t row : group) { if (raw_buffer->IsValid(row)) { running_total += raw_buffer->Value(row); has_data = true; } }
                if (has_data) (void)arr_maker.Append(running_total); else (void)arr_maker.AppendNull();
            } else if (operation == "mean") {
                NativeT running_total = 0; int32_t valid_elements = 0;
                for (int64_t row : group) { if (raw_buffer->IsValid(row)) { running_total += raw_buffer->Value(row); valid_elements++; } }
                if (valid_elements > 0) (void)arr_maker.Append(running_total / valid_elements); else (void)arr_maker.AppendNull();
            } else if (operation == "max") {
                NativeT current_max = 0; bool has_data = false;
                for (int64_t row : group) { 
                    if (raw_buffer->IsValid(row)) { 
                        if (!has_data) { current_max = raw_buffer->Value(row); has_data = true; }
                        else if (raw_buffer->Value(row) > current_max) current_max = raw_buffer->Value(row);
                    }
                }
                if (has_data) (void)arr_maker.Append(current_max); else (void)arr_maker.AppendNull();
            } else if (operation == "min") {
                NativeT current_min = 0; bool has_data = false;
                for (int64_t row : group) { 
                    if (raw_buffer->IsValid(row)) { 
                        if (!has_data) { current_min = raw_buffer->Value(row); has_data = true; }
                        else if (raw_buffer->Value(row) < current_min) current_min = raw_buffer->Value(row);
                    }
                }
                if (has_data) (void)arr_maker.Append(current_min); else (void)arr_maker.AppendNull();
            }
        }
        std::shared_ptr<arrow::Array> finished_buffer;
        (void)arr_maker.Finish(&finished_buffer);
        return std::make_shared<Column>(finished_buffer, (operation == "count") ? DataType::Int32 : type_tag);
    }

    template <typename ArrowT>
    std::shared_ptr<Column> execute_categorical_count(std::shared_ptr<Column> source_col, const std::vector<std::vector<int64_t>>& mapped_groups, const std::string& operation) {
        if (operation != "count") throw std::runtime_error("Aggregation Error: Categorical data only supports count.");
        
        auto raw_buffer = source_col->as<ArrowT>();
        arrow::Int32Builder arr_maker;
        (void)arr_maker.Reserve(mapped_groups.size());
        
        for (const auto& group : mapped_groups) {
            int32_t valid_elements = 0;
            for (int64_t row : group) { if (raw_buffer->IsValid(row)) valid_elements++; }
            (void)arr_maker.Append(valid_elements);
        }
        std::shared_ptr<arrow::Array> finished_buffer;
        (void)arr_maker.Finish(&finished_buffer);
        return std::make_shared<Column>(finished_buffer, DataType::Int32);
    }
}

class GroupedDataFrame {
private:
    const EagerDataFrame& parent_table;
    std::vector<std::string> grouping_identifiers;

public:
    GroupedDataFrame(const EagerDataFrame& parent, std::vector<std::string> keys) 
        : parent_table(parent), grouping_identifiers(std::move(keys)) {}


EagerDataFrame aggregate(const std::vector<std::pair<std::string, std::string>>& computations) {
        EagerDataFrame final_table;
        if (grouping_identifiers.empty())
            throw std::invalid_argument("Aggregation Error: No group key provided.");

        std::vector<std::shared_ptr<Column>> key_cols;
        std::vector<DataType> key_types;
        for (const auto& k : grouping_identifiers) {
            key_cols.push_back(parent_table.fetch_column(k));
            key_types.push_back(key_cols.back()->getType());
        }
        int64_t n = key_cols.front()->size();

        auto stringify_at = [](std::shared_ptr<Column> c, int64_t i) -> std::string {
            DataType dt = c->getType();
            if (dt == DataType::Int32)   { auto a = c->as<arrow::Int32Array>();   return a->IsValid(i) ? std::to_string(a->Value(i)) : "\x01N\x01"; }
            if (dt == DataType::Int64)   { auto a = c->as<arrow::Int64Array>();   return a->IsValid(i) ? std::to_string(a->Value(i)) : "\x01N\x01"; }
            if (dt == DataType::Float32) { auto a = c->as<arrow::FloatArray>();   return a->IsValid(i) ? std::to_string(a->Value(i)) : "\x01N\x01"; }
            if (dt == DataType::Float64) { auto a = c->as<arrow::DoubleArray>();  return a->IsValid(i) ? std::to_string(a->Value(i)) : "\x01N\x01"; }
            if (dt == DataType::Boolean) { auto a = c->as<arrow::BooleanArray>(); return a->IsValid(i) ? (a->Value(i) ? "1" : "0") : "\x01N\x01"; }
            if (dt == DataType::String)  { auto a = c->as<arrow::StringArray>();  return a->IsValid(i) ? a->GetString(i) : "\x01N\x01"; }
            throw std::runtime_error("Group key type unsupported");
        };

        std::unordered_map<std::string, int64_t> bucket_id;
        std::vector<std::vector<int64_t>> hash_groups;
        std::vector<int64_t> first_seen_indices;
        for (int64_t r = 0; r < n; ++r) {
            std::string composite;
            for (size_t k = 0; k < key_cols.size(); ++k) {
                composite += stringify_at(key_cols[k], r);
                composite += '\x02';
            }
            auto it = bucket_id.find(composite);
            if (it == bucket_id.end()) {
                bucket_id[composite] = (int64_t)hash_groups.size();
                hash_groups.emplace_back();
                first_seen_indices.push_back(r);
            }
            hash_groups[bucket_id[composite]].push_back(r);
        }

        for (size_t k = 0; k < grouping_identifiers.size(); ++k) {
            auto col = key_cols[k];
            DataType tag = key_types[k];
            std::shared_ptr<Column> rebuilt;
            if      (tag == DataType::Int32)   rebuilt = engine_utils::gather_by_indices<arrow::Int32Array, arrow::Int32Builder>(col, first_seen_indices, tag);
            else if (tag == DataType::Int64)   rebuilt = engine_utils::gather_by_indices<arrow::Int64Array, arrow::Int64Builder>(col, first_seen_indices, tag);
            else if (tag == DataType::Float32) rebuilt = engine_utils::gather_by_indices<arrow::FloatArray, arrow::FloatBuilder>(col, first_seen_indices, tag);
            else if (tag == DataType::Float64) rebuilt = engine_utils::gather_by_indices<arrow::DoubleArray, arrow::DoubleBuilder>(col, first_seen_indices, tag);
            else if (tag == DataType::Boolean) rebuilt = engine_utils::gather_by_indices<arrow::BooleanArray, arrow::BooleanBuilder>(col, first_seen_indices, tag);
            else if (tag == DataType::String)  rebuilt = engine_utils::gather_by_indices<arrow::StringArray, arrow::StringBuilder>(col, first_seen_indices, tag);
            final_table.insert_mock_column(grouping_identifiers[k], rebuilt);
        }

        for (const auto& request : computations) {
            auto src_col = parent_table.fetch_column(request.first);
            DataType m_dt = src_col->getType();
            std::shared_ptr<Column> computed_col;
            // NEW
if (request.second == "mean") {
    // Forceing ALL mean operations into a DoubleBuilder to prevent integer truncation
    if      (m_dt == DataType::Int32)   computed_col = aggregation_utils::execute_numeric_agg<arrow::Int32Array, arrow::DoubleBuilder, double>(src_col, hash_groups, request.second, DataType::Float64);
    else if (m_dt == DataType::Int64)   computed_col = aggregation_utils::execute_numeric_agg<arrow::Int64Array, arrow::DoubleBuilder, double>(src_col, hash_groups, request.second, DataType::Float64);
    else if (m_dt == DataType::Float32) computed_col = aggregation_utils::execute_numeric_agg<arrow::FloatArray, arrow::DoubleBuilder, double>(src_col, hash_groups, request.second, DataType::Float64);
    else if (m_dt == DataType::Float64) computed_col = aggregation_utils::execute_numeric_agg<arrow::DoubleArray, arrow::DoubleBuilder, double>(src_col, hash_groups, request.second, DataType::Float64);
    else throw std::runtime_error("Aggregation Error: Cannot calculate mean on non-numeric columns.");
} else {
    // then, Normal routing for sum, min, max, count
    if      (m_dt == DataType::Int32)   computed_col = aggregation_utils::execute_numeric_agg<arrow::Int32Array, arrow::Int32Builder, int32_t>(src_col, hash_groups, request.second, m_dt);
    else if (m_dt == DataType::Int64)   computed_col = aggregation_utils::execute_numeric_agg<arrow::Int64Array, arrow::Int64Builder, int64_t>(src_col, hash_groups, request.second, m_dt);
    else if (m_dt == DataType::Float32) computed_col = aggregation_utils::execute_numeric_agg<arrow::FloatArray, arrow::FloatBuilder, float>(src_col, hash_groups, request.second, m_dt);
    else if (m_dt == DataType::Float64) computed_col = aggregation_utils::execute_numeric_agg<arrow::DoubleArray, arrow::DoubleBuilder, double>(src_col, hash_groups, request.second, m_dt);
    else if (m_dt == DataType::String)  computed_col = aggregation_utils::execute_categorical_count<arrow::StringArray>(src_col, hash_groups, request.second);
    else if (m_dt == DataType::Boolean) computed_col = aggregation_utils::execute_categorical_count<arrow::BooleanArray>(src_col, hash_groups, request.second);
    else throw std::runtime_error("Aggregation Error: Unsupported compute type.");
}
            final_table.insert_mock_column(request.first + "_" + request.second, computed_col);
        }
        return final_table;
    }
};

inline GroupedDataFrame EagerDataFrame::group_by(const std::vector<std::string>& keys) const {
    return GroupedDataFrame(*this, keys);
}
inline EagerDataFrame read_parquet(const std::string& location) {
    auto file_res = arrow::io::ReadableFile::Open(location);
    if (!file_res.ok()) throw std::runtime_error("I/O Error: Unable to open Parquet.");

    auto reader_res = parquet::arrow::OpenFile(file_res.ValueOrDie(), arrow::default_memory_pool());
    if (!reader_res.ok()) throw std::runtime_error("I/O Error: Parquet engine initialization failed.");
    std::unique_ptr<parquet::arrow::FileReader> pq_engine = std::move(reader_res.ValueOrDie());

    auto table_read_res = pq_engine->ReadTable();
    if (!table_read_res.ok()) throw std::runtime_error("I/O Error: Parquet parsing failed.");
    std::shared_ptr<arrow::Table> internal_table = table_read_res.ValueOrDie();

    auto flatten_res = internal_table->CombineChunks(arrow::default_memory_pool());
    if (!flatten_res.ok()) throw std::runtime_error("I/O Error: Arrow array chunking failed.");
    auto unified_table = flatten_res.ValueOrDie();

    EagerDataFrame imported_df;
    for (int col_i = 0; col_i < unified_table->num_columns(); ++col_i) {
        std::string header = unified_table->field(col_i)->name();
        auto chunk_zero = unified_table->column(col_i)->chunk(0);

        switch (chunk_zero->type_id()) {
            case arrow::Type::INT32:
                imported_df.insert_mock_column(header, std::make_shared<Column>(chunk_zero, DataType::Int32));
                break;
            case arrow::Type::INT64:
                imported_df.insert_mock_column(header, std::make_shared<Column>(chunk_zero, DataType::Int64));
                break;
            case arrow::Type::FLOAT:
                imported_df.insert_mock_column(header, std::make_shared<Column>(chunk_zero, DataType::Float32));
                break;
            case arrow::Type::DOUBLE:
                imported_df.insert_mock_column(header, std::make_shared<Column>(chunk_zero, DataType::Float64));
                break;
            case arrow::Type::BOOL:
                imported_df.insert_mock_column(header, std::make_shared<Column>(chunk_zero, DataType::Boolean));
                break;
            case arrow::Type::STRING:
                imported_df.insert_mock_column(header, std::make_shared<Column>(chunk_zero, DataType::String));
                break;
            default:
                throw std::runtime_error("Reader Error: Unsupported Arrow type for column '" + header + "'");
        }
    }
    return imported_df;
}

inline EagerDataFrame read_csv(const std::string& location) {
    auto file_res = arrow::io::ReadableFile::Open(location);
    if (!file_res.ok()) throw std::runtime_error("I/O Error: CSV file could not be opened.");

    auto csv_engine_res = arrow::csv::TableReader::Make(
        arrow::io::default_io_context(), file_res.ValueOrDie(),
        arrow::csv::ReadOptions::Defaults(), arrow::csv::ParseOptions::Defaults(), arrow::csv::ConvertOptions::Defaults()
    );
    if (!csv_engine_res.ok()) throw std::runtime_error("I/O Error: CSV Engine failed to boot.");

    auto table_res = csv_engine_res.ValueOrDie()->Read();
    if (!table_res.ok()) throw std::runtime_error("I/O Error: CSV Read pipeline failed.");

    auto flatten_res = table_res.ValueOrDie()->CombineChunks(arrow::default_memory_pool());
    if (!flatten_res.ok()) throw std::runtime_error("I/O Error: Memory defragmentation failed.");
    auto unified_table = flatten_res.ValueOrDie();

    EagerDataFrame imported_df;
    for (int col_i = 0; col_i < unified_table->num_columns(); ++col_i) {
        std::string header = unified_table->field(col_i)->name();
        auto chunk_zero = unified_table->column(col_i)->chunk(0);

        switch (chunk_zero->type_id()) {
            case arrow::Type::INT32:
                imported_df.insert_mock_column(header, std::make_shared<Column>(chunk_zero, DataType::Int32));
                break;
            case arrow::Type::INT64:
                imported_df.insert_mock_column(header, std::make_shared<Column>(chunk_zero, DataType::Int64));
                break;
            case arrow::Type::FLOAT:
                imported_df.insert_mock_column(header, std::make_shared<Column>(chunk_zero, DataType::Float32));
                break;
            case arrow::Type::DOUBLE:
                imported_df.insert_mock_column(header, std::make_shared<Column>(chunk_zero, DataType::Float64));
                break;
            case arrow::Type::BOOL:
                imported_df.insert_mock_column(header, std::make_shared<Column>(chunk_zero, DataType::Boolean));
                break;
            case arrow::Type::STRING:
                imported_df.insert_mock_column(header, std::make_shared<Column>(chunk_zero, DataType::String));
                break;
            default:
                throw std::runtime_error("Reader Error: Unsupported Arrow type for column '" + header + "'");
        }
    }
    return imported_df;
}

}