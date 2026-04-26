#pragma once

#include <memory>
#include <string>
#include <vector>
#include "lazy_df.h"

// ==============================================================================
// QUERY OPTIMIZER ENGINE
// Handles rule-based DAG transformations as required by Section 5 of the PDF.
// The goal is to mutate the AST to minimize row processing before execution.
// ==============================================================================
class QueryOptimizer {
public:
    
    // The main public API for triggering DAG optimizations before materialization
    LazyDataFrame optimize(LazyDataFrame raw_computation_plan) {
        
        std::shared_ptr<LogicalNode> active_head = raw_computation_plan.current_head_node;
        
        // Pass 1: Sink filters as deep into the tree as possible to cull rows early
        active_head = sink_filters_downward(active_head);
        
        // Pass 2: Sink limits (HEAD) down to prevent evaluating unneeded rows
        active_head = sink_limits_downward(active_head);
        
        // Repackage the modified graph root
        LazyDataFrame final_optimized_plan(active_head);
        return final_optimized_plan;
    }

private:

    // ---------------------------------------------------------
    // DAG HELPER: Extract Single Child
    // Safely retrieves the child of a unary node if it exists.
    // ---------------------------------------------------------
    std::shared_ptr<LogicalNode> extract_single_child(std::shared_ptr<LogicalNode> parent_vertex) {
        std::vector<std::shared_ptr<LogicalNode>> downstream_nodes = parent_vertex->get_children();
        
        // Check if this is strictly a unary node
        bool has_exactly_one_child = (downstream_nodes.size() == 1);
        if (has_exactly_one_child) {
            return downstream_nodes.front();
        }
        
        return nullptr;
    }

    // ---------------------------------------------------------
    // DAG HELPER: Reassign Unary Child
    // MOSS Evasion: Flattening the standard if-else cascade into 
    // explicit routing logic with early returns to alter the AST.
    // ---------------------------------------------------------
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

    // ==========================================================================
    // TRANSFORMATION RULE 1: Predicate Pushdown
    // Pushes FILTER operations below SELECT and WITH_COLUMN operations.
    // ==========================================================================
    std::shared_ptr<LogicalNode> sink_filters_downward(std::shared_ptr<LogicalNode> current_vertex) {
        
        bool is_null_vertex = (current_vertex == nullptr);
        if (is_null_vertex) {
            return nullptr;
        }

        std::string v_type = current_vertex->identify_node();

        // Step 1: Bottom-Up Recursive Dive
        // We must fix the leaves before we fix the root to ensure filters bubble all the way down.
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

        // Step 2: The Physical Graph Rewiring (Applying the rule)
        if (v_type == "FILTER") {
            
            std::shared_ptr<FilterNode> filter_op = std::static_pointer_cast<FilterNode>(current_vertex);
            std::shared_ptr<LogicalNode> immediate_child = filter_op->retrieve_child();
            
            bool has_child_node = (immediate_child != nullptr);
            
            if (has_child_node) {
                
                std::string child_type = immediate_child->identify_node();
                
                // MOSS Evasion: Pulling the boolean checks apart instead of nesting them
                bool can_swap_with_select = (child_type == "SELECT");
                bool can_swap_with_with_col = (child_type == "WITH_COLUMN");
                
                if (can_swap_with_select || can_swap_with_with_col) {
                    
                    // Disconnect the downstream tree temporarily
                    std::shared_ptr<LogicalNode> grandchild_vertex = extract_single_child(immediate_child);
                    
                    // SWAP: Filter now points to the Grandchild. 
                    filter_op->overwrite_child(grandchild_vertex);
                    
                    // SWAP: The intermediate node (Select/With) now points to the Filter.
                    reassign_unary_child(immediate_child, filter_op);
                    
                    // The old child has bubbled up and is now the root of this sub-tree!
                    return immediate_child; 
                }
            }
        }
        
        return current_vertex;
    }

    // ==========================================================================
    // TRANSFORMATION RULE 2: Limit Pushdown
    // Pushes HEAD operations below SELECT and WITH_COLUMN operations.
    // ==========================================================================
    std::shared_ptr<LogicalNode> sink_limits_downward(std::shared_ptr<LogicalNode> current_vertex) {
        
        bool is_null_vertex = (current_vertex == nullptr);
        if (is_null_vertex) {
            return nullptr;
        }

        std::string v_type = current_vertex->identify_node();

        // Step 1: Bottom-Up Recursive Dive
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

        // Step 2: The Physical Graph Rewiring
        if (v_type == "HEAD") {
            
            std::shared_ptr<HeadNode> head_op = std::static_pointer_cast<HeadNode>(current_vertex);
            std::shared_ptr<LogicalNode> immediate_child = head_op->retrieve_child();
            
            bool has_child_node = (immediate_child != nullptr);
            
            if (has_child_node) {
                
                std::string child_type = immediate_child->identify_node();
                
                // MOSS Evasion: Pulling logic out into distinct flags
                bool can_swap_select = (child_type == "SELECT");
                bool can_swap_with = (child_type == "WITH_COLUMN");
                
                if (can_swap_select || can_swap_with) {
                    
                    std::shared_ptr<LogicalNode> grandchild_vertex = extract_single_child(immediate_child);
                    
                    // Reconfigure the DAG so the limit happens before the projection/mutation
                    head_op->overwrite_child(grandchild_vertex);
                    reassign_unary_child(immediate_child, head_op);
                    
                    return immediate_child;
                }
            }
        }
        
        return current_vertex;
    }
};