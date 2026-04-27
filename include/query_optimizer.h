#pragma once

#include <memory>
#include <string>
#include <vector>
#include "lazy_df.h"
#include <unordered_set>

#include <arrow/io/api.h>
#include <arrow/csv/api.h>
#include <parquet/arrow/reader.h>

#include <fstream>
#include <sstream> 


namespace dataframelib {


    inline void collect_col_refs(const std::shared_ptr<Exp>& e, std::unordered_set<std::string>& out) {
    if (!e) return;
    if (auto c = std::dynamic_pointer_cast<ColRef>(e))    { out.insert(c->name()); return; }
    if (auto b = std::dynamic_pointer_cast<BinaryOp>(e))  { collect_col_refs(b->lhs_node, out); collect_col_refs(b->rhs_node, out); return; }
    if (auto u = std::dynamic_pointer_cast<UnaryOp>(e))   { collect_col_refs(u->child_operand, out); return; }
    if (auto m = std::dynamic_pointer_cast<MethodOp>(e))  { collect_col_refs(m->base_target, out); return; }
}


inline std::shared_ptr<Exp> fold_constants(std::shared_ptr<Exp> e) {
    if (!e) return e;
    if (auto b = std::dynamic_pointer_cast<BinaryOp>(e)) {
        b->lhs_node = fold_constants(b->lhs_node);
        b->rhs_node = fold_constants(b->rhs_node);
        auto lc = std::dynamic_pointer_cast<ConstantNode>(b->lhs_node);
        auto rc = std::dynamic_pointer_cast<ConstantNode>(b->rhs_node);
        if (lc && rc) {
            // if both children are literals then I evaluated them and replaced them with a single constant
            DataFrameColumns empty;
            auto col = b->evaluate(empty);
            DataType dt = col->getType();
            std::string s;
            if (dt == DataType::Int32)        s = std::to_string(col->as<arrow::Int32Array>()->Value(0));
            else if (dt == DataType::Int64)   s = std::to_string(col->as<arrow::Int64Array>()->Value(0));
            else if (dt == DataType::Float32) s = std::to_string(col->as<arrow::FloatArray>()->Value(0));
            else if (dt == DataType::Float64) s = std::to_string(col->as<arrow::DoubleArray>()->Value(0));
            else if (dt == DataType::Boolean) s = col->as<arrow::BooleanArray>()->Value(0) ? "true" : "false";
            else if (dt == DataType::String)  s = col->as<arrow::StringArray>()->GetString(0);
            return std::make_shared<ConstantNode>(dt, s);
        }
        return b;
    }
    if (auto u = std::dynamic_pointer_cast<UnaryOp>(e))   { u->child_operand = fold_constants(u->child_operand); return u; }
    if (auto m = std::dynamic_pointer_cast<MethodOp>(e))  { m->base_target   = fold_constants(m->base_target);   return m; }
    return e;
}


inline std::shared_ptr<Exp> simplify_expr(std::shared_ptr<Exp> e) {
    if (!e) return e;
    if (auto b = std::dynamic_pointer_cast<BinaryOp>(e)) {
        b->lhs_node = simplify_expr(b->lhs_node);
        b->rhs_node = simplify_expr(b->rhs_node);
        auto lc = std::dynamic_pointer_cast<ConstantNode>(b->lhs_node);
        auto rc = std::dynamic_pointer_cast<ConstantNode>(b->rhs_node);
        auto is_zero = [](const std::shared_ptr<ConstantNode>& c) {
            if (!c) return false;
            std::string s = c->to_string();
            return s == "0" || s == "0.000000" || s == "0.0";
        };
        auto is_one = [](const std::shared_ptr<ConstantNode>& c) {
            if (!c) return false;
            std::string s = c->to_string();
            return s == "1" || s == "1.000000" || s == "1.0";
        };
        auto is_true  = [](const std::shared_ptr<ConstantNode>& c) { return c && c->to_string() == "true";  };
        auto is_false = [](const std::shared_ptr<ConstantNode>& c) { return c && c->to_string() == "false"; };

        const std::string& op = b->operation_token;
        if (op == "+" && is_zero(rc)) return b->lhs_node;
        if (op == "+" && is_zero(lc)) return b->rhs_node;
        if (op == "-" && is_zero(rc)) return b->lhs_node;
        if (op == "*" && is_one(rc))  return b->lhs_node;
        if (op == "*" && is_one(lc))  return b->rhs_node;
        if (op == "/" && is_one(rc))  return b->lhs_node;
        if (op == "&" && is_true(rc)) return b->lhs_node;
        if (op == "&" && is_true(lc)) return b->rhs_node;
        if (op == "|" && is_false(rc))return b->lhs_node;
        if (op == "|" && is_false(lc))return b->rhs_node;
        return b;
    }
    if (auto u = std::dynamic_pointer_cast<UnaryOp>(e))   { u->child_operand = simplify_expr(u->child_operand); return u; }
    if (auto m = std::dynamic_pointer_cast<MethodOp>(e))  { m->base_target   = simplify_expr(m->base_target);   return m; }
    return e;
}


// code for Flattening somthing like "(a & b) & c" into [a, b, c]
inline void split_conjuncts(const std::shared_ptr<Exp>& e,
                            std::vector<std::shared_ptr<Exp>>& out) {
    auto b = std::dynamic_pointer_cast<BinaryOp>(e);
    if (b && b->operation_token == "&") {
        split_conjuncts(b->lhs_node, out);
        split_conjuncts(b->rhs_node, out);
    } else if (e) {
        out.push_back(e);
    }
}

inline std::shared_ptr<Exp> conjoin(const std::vector<std::shared_ptr<Exp>>& parts) {
    if (parts.empty()) return nullptr;
    auto out = parts[0];
    for (size_t i = 1; i < parts.size(); ++i)
        out = std::make_shared<BinaryOp>(out, parts[i], "&");
    return out;
}

inline std::unordered_set<std::string> scan_schema(const std::string& path, bool is_parquet) {
    std::unordered_set<std::string> out;

    if (is_parquet) {
        auto file_res = arrow::io::ReadableFile::Open(path);
        if (!file_res.ok()) return out;
        auto r = parquet::arrow::OpenFile(file_res.ValueOrDie(), arrow::default_memory_pool());
        if (!r.ok()) return out;
        std::shared_ptr<arrow::Schema> sch;
        if (!r.ValueOrDie()->GetSchema(&sch).ok()) return out;
        for (int i = 0; i < sch->num_fields(); ++i) out.insert(sch->field(i)->name());
        return out;
    }


    std::ifstream f(path);
    if (!f.is_open()) return out;
    std::string header;
    if (!std::getline(f, header)) return out;
    if (!header.empty() && header.back() == '\r') header.pop_back();

    std::stringstream ss(header);
    std::string col;
    while (std::getline(ss, col, ',')) {
        // code for stripping surrounding whitespace and optional quotes
        size_t a = col.find_first_not_of(" \t\"");
        size_t b = col.find_last_not_of(" \t\"");
        if (a == std::string::npos) continue;
        out.insert(col.substr(a, b - a + 1));
    }
    return out;
}

// here, i computed the set of columns which a sub-DAG produces.
inline std::unordered_set<std::string> output_columns(const std::shared_ptr<LogicalNode>& v) {
    if (!v) return {};
    std::string t = v->identify_node();
    if (t == "SCAN_CSV")     return scan_schema(std::static_pointer_cast<ScanNode>(v)->get_path(), false);
    if (t == "SCAN_PARQUET") return scan_schema(std::static_pointer_cast<ScanParquetNode>(v)->get_path(), true);
    if (t == "FILTER")       return output_columns(std::static_pointer_cast<FilterNode>(v)->retrieve_child());
    if (t == "SORT")         return output_columns(std::static_pointer_cast<SortNode>(v)->retrieve_child());
    if (t == "HEAD")         return output_columns(std::static_pointer_cast<HeadNode>(v)->retrieve_child());
    if (t == "WITH_COLUMN") {
        auto w = std::static_pointer_cast<WithColumnNode>(v);
        auto s = output_columns(w->retrieve_child());
        s.insert(w->get_added_name());
        return s;
    }
    if (t == "SELECT") {
        auto cols = std::static_pointer_cast<SelectNode>(v)->get_columns();
        return {cols.begin(), cols.end()};
    }
    if (t == "AGGREGATE") {
        return output_columns(std::static_pointer_cast<AggregateNode>(v)->retrieve_child());
    }
    if (t == "JOIN") {
        auto j = std::static_pointer_cast<JoinNode>(v);
        auto L = output_columns(j->fetch_left_child());
        auto R = output_columns(j->fetch_right_child());
        L.insert(R.begin(), R.end());
        return L;
    }
    return {};
}


// this is the code for a quesry optimizer engine which handles rule-based DAG transformations to minimize row processing before execution.
class QueryOptimizer {
public:
    
    //this is the  public API for triggering DAG optimizations before materialization
    /**
 * @brief Applies rule-based algebraic transformations to a logical query plan.
 * Transforms include predicate pushdown, limit pushdown, projection pushdown, and constant folding.
 * @param raw_computation_plan The unoptimized, user-defined lazy execution plan.
 * @return A new LazyDataFrame containing the highly optimized execution DAG.
 */

    LazyDataFrame optimize(LazyDataFrame raw_computation_plan) {
        std::shared_ptr<LogicalNode> active_head = raw_computation_plan.current_head_node;
        active_head = rewrite_expressions(active_head); 
        active_head = sink_filters_downward(active_head);
        active_head = sink_limits_downward(active_head);
        active_head = sink_projections_downward(active_head); // NEW: Projection pushdown
        return LazyDataFrame(active_head);
    }

private:

    std::shared_ptr<LogicalNode> rewrite_expressions(std::shared_ptr<LogicalNode> v) {
        if (!v) return v;
        std::string t = v->identify_node();
        if (t == "FILTER") {
            auto f = std::static_pointer_cast<FilterNode>(v);
            auto p = simplify_expr(fold_constants(f->get_predicate()));
            auto nf = std::make_shared<FilterNode>(f->retrieve_child(), p);
            v = nf;
        } else if (t == "WITH_COLUMN") {
            auto w = std::static_pointer_cast<WithColumnNode>(v);
            auto e = simplify_expr(fold_constants(w->get_expression()));
            auto nw = std::make_shared<WithColumnNode>(w->retrieve_child(), w->get_added_name(), e);
            v = nw;
        }
        // recurstion into children
        if (t == "JOIN") {
            auto j = std::static_pointer_cast<JoinNode>(v);
            j->update_left_branch(rewrite_expressions(j->fetch_left_child()));
            j->update_right_branch(rewrite_expressions(j->fetch_right_child()));
        } else {
            auto child = extract_single_child(v);
            if (child) reassign_unary_child(v, rewrite_expressions(child));
        }
        return v;
    }



    // if the child of a unary node exists then returning it
    std::shared_ptr<LogicalNode> extract_single_child(std::shared_ptr<LogicalNode> parent_vertex) {
        std::vector<std::shared_ptr<LogicalNode>> downstream_nodes = parent_vertex->get_children();
        

        bool has_exactly_one_child = (downstream_nodes.size() == 1);
        if (has_exactly_one_child) {
            return downstream_nodes.front();
        }
        
        return nullptr;
    }

    void reassign_unary_child(std::shared_ptr<LogicalNode> parent_vertex, std::shared_ptr<LogicalNode> replacement_child) {
        
        std::string vertex_id = parent_vertex->identify_node();
        
        if (vertex_id == "FILTER") {
            std::static_pointer_cast<FilterNode>(parent_vertex)->overwrite_child(replacement_child);
            return;
        }
        if (vertex_id == "SELECT") {
            std::static_pointer_cast<SelectNode>(parent_vertex)->overwrite_child(replacement_child);
            return;
        }
        if (vertex_id == "WITH_COLUMN") {
            std::static_pointer_cast<WithColumnNode>(parent_vertex)->overwrite_child(replacement_child);
            return;
        }
        if (vertex_id == "SORT") {
            std::static_pointer_cast<SortNode>(parent_vertex)->overwrite_child(replacement_child);
            return;
        }
        if (vertex_id == "HEAD") {
            std::static_pointer_cast<HeadNode>(parent_vertex)->overwrite_child(replacement_child);
            return;
        }
        if (vertex_id == "AGGREGATE") {
            std::static_pointer_cast<AggregateNode>(parent_vertex)->overwrite_child(replacement_child);
            return;
        }
    }


    // push the filter operations below select and with_column ops
    std::shared_ptr<LogicalNode> sink_filters_downward(std::shared_ptr<LogicalNode> current_vertex) {
        
        bool is_null_vertex = (current_vertex == nullptr);
        if (is_null_vertex) {
            return nullptr;
        }

        std::string v_type = current_vertex->identify_node();

        if (v_type == "JOIN") {
            std::shared_ptr<JoinNode> join_op = std::static_pointer_cast<JoinNode>(current_vertex);
            
            std::shared_ptr<LogicalNode> optimized_left = sink_filters_downward(join_op->fetch_left_child());
            std::shared_ptr<LogicalNode> optimized_right = sink_filters_downward(join_op->fetch_right_child());
            
            join_op->update_left_branch(optimized_left);
            join_op->update_right_branch(optimized_right);
        } 
        else {
            std::shared_ptr<LogicalNode> single_child = extract_single_child(current_vertex);
            if (single_child != nullptr) {
                std::shared_ptr<LogicalNode> optimized_child = sink_filters_downward(single_child);
                reassign_unary_child(current_vertex, optimized_child);
            }
        }

        if (v_type == "FILTER") {
            auto filter_op = std::static_pointer_cast<FilterNode>(current_vertex);
            auto immediate_child = filter_op->retrieve_child();
            if (!immediate_child) return current_vertex;

            std::string ct = immediate_child->identify_node();

            if (ct == "JOIN") {
                auto join_op = std::static_pointer_cast<JoinNode>(immediate_child);

                std::vector<std::shared_ptr<Exp>> conjuncts;
                split_conjuncts(filter_op->get_predicate(), conjuncts);

                auto left_schema  = output_columns(join_op->fetch_left_child());
                auto right_schema = output_columns(join_op->fetch_right_child());


                auto join_keys = join_op->fetch_join_keys();
                std::unordered_set<std::string> join_key_set(join_keys.begin(), join_keys.end());

                std::vector<std::shared_ptr<Exp>> left_pushed, right_pushed, residual;
                for (auto& c : conjuncts) {
                    std::unordered_set<std::string> refs;
                    collect_col_refs(c, refs);

                    bool touches_join_key = false;
                    for (const auto& r : refs) {
                        if (join_key_set.count(r)) { touches_join_key = true; break; }
                    }
                    if (touches_join_key || refs.empty()) {
                        residual.push_back(c);
                        continue;
                    }

                    bool all_left  = true;
                    bool all_right = true;
                    for (const auto& r : refs) {
                        if (!left_schema.count(r))  all_left  = false;
                        if (!right_schema.count(r)) all_right = false;
                    }
                    if      (all_left)  left_pushed.push_back(c);
                    else if (all_right) right_pushed.push_back(c);
                    else                residual.push_back(c);
                }

                if (!left_pushed.empty()) {
                    auto lf = std::make_shared<FilterNode>(join_op->fetch_left_child(), conjoin(left_pushed));
                    join_op->update_left_branch(sink_filters_downward(lf));
                }
                if (!right_pushed.empty()) {
                    auto rf = std::make_shared<FilterNode>(join_op->fetch_right_child(), conjoin(right_pushed));
                    join_op->update_right_branch(sink_filters_downward(rf));
                }

                if (residual.empty()) return join_op;
                return std::make_shared<FilterNode>(join_op, conjoin(residual));
            }

            std::unordered_set<std::string> needed;
            collect_col_refs(filter_op->get_predicate(), needed);

            bool can_swap = false;
            if (ct == "SELECT") {
                can_swap = true;
            } else if (ct == "WITH_COLUMN") {
                auto wc = std::static_pointer_cast<WithColumnNode>(immediate_child);
                can_swap = (needed.find(wc->get_added_name()) == needed.end());
            }

            if (can_swap) {
                auto grandchild = extract_single_child(immediate_child);
                filter_op->overwrite_child(grandchild);
                reassign_unary_child(immediate_child, filter_op);
                return immediate_child;
            }
        }
        
    return current_vertex;
    }


    // push head operation below select and with_column ops
    std::shared_ptr<LogicalNode> sink_limits_downward(std::shared_ptr<LogicalNode> current_vertex) {
        
        bool is_null_vertex = (current_vertex == nullptr);
        if (is_null_vertex) {
            return nullptr;
        }

        std::string v_type = current_vertex->identify_node();

        // code for Bottom-Up Recursive Dive
        if (v_type == "JOIN") {
            std::shared_ptr<JoinNode> join_op = std::static_pointer_cast<JoinNode>(current_vertex);
            
            std::shared_ptr<LogicalNode> pushed_left = sink_limits_downward(join_op->fetch_left_child());
            std::shared_ptr<LogicalNode> pushed_right = sink_limits_downward(join_op->fetch_right_child());
            
            join_op->update_left_branch(pushed_left);
            join_op->update_right_branch(pushed_right);
        } 
        else {
            std::shared_ptr<LogicalNode> single_child = extract_single_child(current_vertex);
            if (single_child != nullptr) {
                std::shared_ptr<LogicalNode> pushed_child = sink_limits_downward(single_child);
                reassign_unary_child(current_vertex, pushed_child);
            }
        }

        // The Physical Graph Rewiring
        if (v_type == "HEAD") {
            
            std::shared_ptr<HeadNode> head_op = std::static_pointer_cast<HeadNode>(current_vertex);
            std::shared_ptr<LogicalNode> immediate_child = head_op->retrieve_child();
            
            bool has_child_node = (immediate_child != nullptr);
            
            if (has_child_node) {
                
                std::string child_type = immediate_child->identify_node();
                
                // Useing boolean flags 
                bool can_swap_select = (child_type == "SELECT");
                bool can_swap_with = (child_type == "WITH_COLUMN");
                
                if (can_swap_select || can_swap_with) {
                    
                    std::shared_ptr<LogicalNode> grandchild_vertex = extract_single_child(immediate_child);
                    
                    // I Reconfigured the DAG so the limit happens before the projection/mutation
                    head_op->overwrite_child(grandchild_vertex);
                    reassign_unary_child(immediate_child, head_op);
                    
                    return immediate_child;
                }
            }
        }
        
        return current_vertex;
    }

    // NEW: Push SELECT operations downward through Filter, WithColumn, Join, and Head ops
    std::shared_ptr<LogicalNode> sink_projections_downward(std::shared_ptr<LogicalNode> current_vertex) {
        if (!current_vertex) return nullptr;

        std::string v_type = current_vertex->identify_node();

        // 1. Bottom-up dive
        if (v_type == "JOIN") {
            auto join_op = std::static_pointer_cast<JoinNode>(current_vertex);
            join_op->update_left_branch(sink_projections_downward(join_op->fetch_left_child()));
            join_op->update_right_branch(sink_projections_downward(join_op->fetch_right_child()));
        } else {
            auto single_child = extract_single_child(current_vertex);
            if (single_child) {
                reassign_unary_child(current_vertex, sink_projections_downward(single_child));
            }
        }

        // 2. Physical DAG Rewiring for Projection Pushdown
        if (v_type == "SELECT") {
            auto select_op = std::static_pointer_cast<SelectNode>(current_vertex);
            auto immediate_child = select_op->retrieve_child();
            if (!immediate_child) return current_vertex;

            std::string ct = immediate_child->identify_node();

            if (ct == "FILTER") {
                auto filter_op = std::static_pointer_cast<FilterNode>(immediate_child);
                std::unordered_set<std::string> needed;
                collect_col_refs(filter_op->get_predicate(), needed);
                
                auto cols = select_op->get_columns();
                for (const auto& c : cols) needed.insert(c);

                auto grandchild_schema = output_columns(filter_op->retrieve_child());
                if (needed.size() < grandchild_schema.size()) {
                    std::vector<std::string> combined_cols(needed.begin(), needed.end());
                    auto pushed_select = std::make_shared<SelectNode>(filter_op->retrieve_child(), combined_cols);
                    pushed_select = std::static_pointer_cast<SelectNode>(sink_projections_downward(pushed_select));
                    filter_op->overwrite_child(pushed_select);
                    
                    if (cols.size() < needed.size()) return select_op;
                    return filter_op;
                }
            } else if (ct == "WITH_COLUMN") {
                auto wc_op = std::static_pointer_cast<WithColumnNode>(immediate_child);
                std::unordered_set<std::string> needed;
                collect_col_refs(wc_op->get_expression(), needed);
                
                auto cols = select_op->get_columns();
                bool requires_added = false;
                for (const auto& c : cols) {
                    if (c == wc_op->get_added_name()) requires_added = true;
                    else needed.insert(c);
                }

                auto grandchild_schema = output_columns(wc_op->retrieve_child());
                if (needed.size() < grandchild_schema.size()) {
                    std::vector<std::string> combined_cols(needed.begin(), needed.end());
                    auto pushed_select = std::make_shared<SelectNode>(wc_op->retrieve_child(), combined_cols);
                    pushed_select = std::static_pointer_cast<SelectNode>(sink_projections_downward(pushed_select));
                    wc_op->overwrite_child(pushed_select);

                    if (requires_added && combined_cols.size() == (cols.size() - (requires_added ? 1 : 0))) return wc_op;
return select_op;
                }
            } else if (ct == "JOIN") {
                auto join_op = std::static_pointer_cast<JoinNode>(immediate_child);
                auto join_keys = join_op->fetch_join_keys();
                std::unordered_set<std::string> needed_left(join_keys.begin(), join_keys.end());
                std::unordered_set<std::string> needed_right(join_keys.begin(), join_keys.end());
                
                auto left_schema = output_columns(join_op->fetch_left_child());
                auto right_schema = output_columns(join_op->fetch_right_child());

                auto cols = select_op->get_columns();
                for (const auto& c : cols) {
                    if (left_schema.count(c)) needed_left.insert(c);
                    if (right_schema.count(c)) needed_right.insert(c);
                }

                bool optimized = false;
                if (needed_left.size() < left_schema.size()) {
                    auto left_sel = std::make_shared<SelectNode>(join_op->fetch_left_child(), std::vector<std::string>(needed_left.begin(), needed_left.end()));
                    join_op->update_left_branch(sink_projections_downward(left_sel));
                    optimized = true;
                }
                if (needed_right.size() < right_schema.size()) {
                    auto right_sel = std::make_shared<SelectNode>(join_op->fetch_right_child(), std::vector<std::string>(needed_right.begin(), needed_right.end()));
                    join_op->update_right_branch(sink_projections_downward(right_sel));
                    optimized = true;
                }

                if (optimized) {
                    if (cols.size() < needed_left.size() + needed_right.size()) return select_op;
                    return join_op;
                }
            } else if (ct == "HEAD") {
                auto head_op = std::static_pointer_cast<HeadNode>(immediate_child);
                auto grandchild = head_op->retrieve_child();
                
                select_op->overwrite_child(grandchild);
                auto pushed_select = sink_projections_downward(select_op);
                head_op->overwrite_child(pushed_select);
                
                return head_op;
            }
        }
        return current_vertex;
    }
};



}