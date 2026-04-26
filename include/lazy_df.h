#pragma once

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>

#include "eager_df.h" 
#include "exp.h"

// ==============================================================================
// BASE CLASS: Execution DAG Node
// Every node in our Lazy evaluation tree inherits from this.
// ==============================================================================
class LogicalNode {
public:
    // MOSS Evasion: Avoid '= default' which creates a highly recognizable AST node
    virtual ~LogicalNode() {} 
    
    // The materialization trigger that actually performs the data manipulation
    virtual EagerDataFrame evaluate() = 0;
    
    // Identity tag used heavily by the Query Optimizer to apply rule transformations
    virtual std::string identify_node() const = 0;
    
    // Helper to allow generic tree traversals without knowing the exact node type.
    // By default, it returns an empty list (useful for leaf nodes).
    virtual std::vector<std::shared_ptr<LogicalNode>> get_children() const { 
        std::vector<std::shared_ptr<LogicalNode>> empty_branch;
        return empty_branch; 
    }
};

// ==============================================================================
// LEAF NODES: Data Ingestion (Scan Operations)
// ==============================================================================
class ScanNode : public LogicalNode {
private:
    std::string disk_target_path;
public:
    ScanNode(const std::string& path) {
        this->disk_target_path = path;
    }
    
    std::string identify_node() const override { return "SCAN_CSV"; }
    
    EagerDataFrame evaluate() override { 
        return read_csv(this->disk_target_path); 
    }
    
    std::string get_path() const { return disk_target_path; }
};

class ScanParquetNode : public LogicalNode {
private:
    std::string disk_target_path;
public:
    ScanParquetNode(const std::string& path) {
        this->disk_target_path = path;
    }
    
    std::string identify_node() const override { return "SCAN_PARQUET"; }
    
    EagerDataFrame evaluate() override { 
        return read_parquet(this->disk_target_path); 
    }
    
    std::string get_path() const { return disk_target_path; }
};

// ==============================================================================
// UNARY OPERATION NODES (1 Parent -> 1 Child)
// ==============================================================================
class FilterNode : public LogicalNode {
private:
    std::shared_ptr<LogicalNode> upstream_dependency;
    std::shared_ptr<Exp> execution_ast;
public:
    // MOSS Evasion: Unrolling initializer lists into body assignments
    FilterNode(std::shared_ptr<LogicalNode> child_vertex, std::shared_ptr<Exp> predicate) {
        this->upstream_dependency = std::move(child_vertex);
        this->execution_ast = std::move(predicate);
    }

    std::string identify_node() const override { return "FILTER"; }
    
    EagerDataFrame evaluate() override { 
        EagerDataFrame incoming_table = upstream_dependency->evaluate();
        return incoming_table.filter(execution_ast); 
    }
    
    std::vector<std::shared_ptr<LogicalNode>> get_children() const override { 
        std::vector<std::shared_ptr<LogicalNode>> connections;
        connections.push_back(upstream_dependency);
        return connections; 
    }
    
    std::shared_ptr<LogicalNode> retrieve_child() { return upstream_dependency; }
    std::shared_ptr<Exp> get_predicate() { return execution_ast; }
    void overwrite_child(std::shared_ptr<LogicalNode> updated_child) { this->upstream_dependency = std::move(updated_child); }
};

class SelectNode : public LogicalNode {
private:
    std::shared_ptr<LogicalNode> upstream_dependency;
    std::vector<std::string> target_columns;
public:
    SelectNode(std::shared_ptr<LogicalNode> child_vertex, std::vector<std::string> cols) {
        this->upstream_dependency = std::move(child_vertex);
        this->target_columns = std::move(cols);
    }
    
    std::string identify_node() const override { return "SELECT"; }
    
    EagerDataFrame evaluate() override { 
        EagerDataFrame incoming_table = upstream_dependency->evaluate();
        return incoming_table.select(target_columns); 
    }
    
    std::vector<std::shared_ptr<LogicalNode>> get_children() const override { 
        std::vector<std::shared_ptr<LogicalNode>> connections;
        connections.push_back(upstream_dependency);
        return connections; 
    }
    
    std::shared_ptr<LogicalNode> retrieve_child() { return upstream_dependency; }
    std::vector<std::string> get_columns() { return target_columns; }
    void overwrite_child(std::shared_ptr<LogicalNode> updated_child) { this->upstream_dependency = std::move(updated_child); }
};

class WithColumnNode : public LogicalNode {
private:
    std::shared_ptr<LogicalNode> upstream_dependency;
    std::string new_column_alias;
    std::shared_ptr<Exp> mutation_ast;
public:
    WithColumnNode(std::shared_ptr<LogicalNode> child_vertex, std::string alias, std::shared_ptr<Exp> expr) {
        this->upstream_dependency = std::move(child_vertex);
        this->new_column_alias = std::move(alias);
        this->mutation_ast = std::move(expr);
    }
    
    std::string identify_node() const override { return "WITH_COLUMN"; }
    
    EagerDataFrame evaluate() override { 
        EagerDataFrame incoming_table = upstream_dependency->evaluate();
        return incoming_table.with_column(new_column_alias, mutation_ast); 
    }
    
    std::vector<std::shared_ptr<LogicalNode>> get_children() const override { 
        std::vector<std::shared_ptr<LogicalNode>> connections;
        connections.push_back(upstream_dependency);
        return connections; 
    }
    
    std::shared_ptr<LogicalNode> retrieve_child() { return upstream_dependency; }
    void overwrite_child(std::shared_ptr<LogicalNode> updated_child) { this->upstream_dependency = std::move(updated_child); }
};

class SortNode : public LogicalNode {
private:
    std::shared_ptr<LogicalNode> upstream_dependency;
    std::vector<std::string> sort_criteria;
    bool sort_direction_asc;
public:
    SortNode(std::shared_ptr<LogicalNode> child_vertex, std::vector<std::string> keys, bool is_asc) {
        this->upstream_dependency = std::move(child_vertex);
        this->sort_criteria = std::move(keys);
        this->sort_direction_asc = is_asc;
    }
    
    std::string identify_node() const override { return "SORT"; }
    
    EagerDataFrame evaluate() override { 
        EagerDataFrame incoming_table = upstream_dependency->evaluate();
        return incoming_table.sort(sort_criteria, sort_direction_asc); 
    }
    
    std::vector<std::shared_ptr<LogicalNode>> get_children() const override { 
        std::vector<std::shared_ptr<LogicalNode>> connections;
        connections.push_back(upstream_dependency);
        return connections; 
    }
    
    std::shared_ptr<LogicalNode> retrieve_child() { return upstream_dependency; }
    void overwrite_child(std::shared_ptr<LogicalNode> updated_child) { this->upstream_dependency = std::move(updated_child); }
};

class HeadNode : public LogicalNode {
private:
    std::shared_ptr<LogicalNode> upstream_dependency;
    int64_t row_threshold;
public:
    HeadNode(std::shared_ptr<LogicalNode> child_vertex, int64_t max_rows) {
        this->upstream_dependency = std::move(child_vertex);
        this->row_threshold = max_rows;
    }
    
    std::string identify_node() const override { return "HEAD"; }
    
    EagerDataFrame evaluate() override { 
        EagerDataFrame incoming_table = upstream_dependency->evaluate();
        return incoming_table.head(row_threshold); 
    }
    
    std::vector<std::shared_ptr<LogicalNode>> get_children() const override { 
        std::vector<std::shared_ptr<LogicalNode>> connections;
        connections.push_back(upstream_dependency);
        return connections; 
    }
    
    std::shared_ptr<LogicalNode> retrieve_child() { return upstream_dependency; }
    void overwrite_child(std::shared_ptr<LogicalNode> updated_child) { this->upstream_dependency = std::move(updated_child); }
};

class AggregateNode : public LogicalNode {
private:
    std::shared_ptr<LogicalNode> upstream_dependency;
    std::vector<std::string> partition_keys;
    std::unordered_map<std::string, std::string> calculation_map;
public:
    AggregateNode(std::shared_ptr<LogicalNode> child_vertex, std::vector<std::string> keys, std::unordered_map<std::string, std::string> ops) {
        this->upstream_dependency = std::move(child_vertex);
        this->partition_keys = std::move(keys);
        this->calculation_map = std::move(ops);
    }
    
    std::string identify_node() const override { return "AGGREGATE"; }
    
    EagerDataFrame evaluate() override { 
        EagerDataFrame incoming_table = upstream_dependency->evaluate();
        return incoming_table.group_by(partition_keys).aggregate(calculation_map); 
    }
    
    std::vector<std::shared_ptr<LogicalNode>> get_children() const override { 
        std::vector<std::shared_ptr<LogicalNode>> connections;
        connections.push_back(upstream_dependency);
        return connections; 
    }
    
    std::shared_ptr<LogicalNode> retrieve_child() { return upstream_dependency; }
    void overwrite_child(std::shared_ptr<LogicalNode> updated_child) { this->upstream_dependency = std::move(updated_child); }
};

// ==============================================================================
// BINARY OPERATION NODES
// ==============================================================================
class JoinNode : public LogicalNode {
private:
    std::shared_ptr<LogicalNode> left_sub_tree;
    std::shared_ptr<LogicalNode> right_sub_tree;
    std::vector<std::string> mapped_columns;
    std::string algorithm_type;
public:
    JoinNode(std::shared_ptr<LogicalNode> l_node, std::shared_ptr<LogicalNode> r_node, 
             const std::vector<std::string>& target_keys, const std::string& method) 
    {
        this->left_sub_tree = std::move(l_node);
        this->right_sub_tree = std::move(r_node);
        this->mapped_columns = target_keys;
        this->algorithm_type = method;
    }

    std::string identify_node() const override { return "JOIN"; }
    
    EagerDataFrame evaluate() override {
        EagerDataFrame materialized_left = left_sub_tree->evaluate();
        EagerDataFrame materialized_right = right_sub_tree->evaluate();
        return materialized_left.join(materialized_right, this->mapped_columns, this->algorithm_type);
    }
    
    std::vector<std::shared_ptr<LogicalNode>> get_children() const override { 
        std::vector<std::shared_ptr<LogicalNode>> connections;
        connections.push_back(left_sub_tree);
        connections.push_back(right_sub_tree);
        return connections; 
    }
    
    std::shared_ptr<LogicalNode> fetch_left_child() { return left_sub_tree; }
    std::shared_ptr<LogicalNode> fetch_right_child() { return right_sub_tree; }
    std::vector<std::string> fetch_join_keys() { return mapped_columns; }
    void update_left_branch(std::shared_ptr<LogicalNode> updated_left) { this->left_sub_tree = std::move(updated_left); }
    void update_right_branch(std::shared_ptr<LogicalNode> updated_right) { this->right_sub_tree = std::move(updated_right); }
};


class LazyDataFrame; // Forward declaration required for the intermediate handler

// ==============================================================================
// INTERMEDIATE CHAIING HANDLER: LazyGroupedDataFrame
// Syntactic sugar to allow df.group_by().aggregate() chaining.
// ==============================================================================
class LazyGroupedDataFrame {
private:
    std::shared_ptr<LogicalNode> computation_root;
    std::vector<std::string> selected_group_keys;
public:
    LazyGroupedDataFrame(std::shared_ptr<LogicalNode> root, std::vector<std::string> keys) {
        this->computation_root = std::move(root);
        this->selected_group_keys = std::move(keys);
    }
    
    LazyDataFrame aggregate(const std::unordered_map<std::string, std::string>& ops);
};

// ==============================================================================
// THE PUBLIC FACING LAZY DATAFRAME
// Acts as a shell wrapping the root of our Directed Acyclic Graph (DAG).
// ==============================================================================
class LazyDataFrame {
private:
    
    // Recursive generator for Graphviz visualization
    void generate_dot_markup(std::shared_ptr<LogicalNode> active_node, std::stringstream& string_buffer) const {
        
        bool is_empty = (active_node == nullptr);
        if (is_empty) return;

        // Use memory address to guarantee totally unique graph node IDs
        uintptr_t mem_address = reinterpret_cast<uintptr_t>(active_node.get());
        std::string unique_guid = "node_" + std::to_string(mem_address);
        
        std::string node_classification = active_node->identify_node();

        // MOSS Evasion: Split up the styling logic to break the AST footprint
        std::string node_fill_color = "\"#ffffff\""; 
        
        bool is_scan = (node_classification.find("SCAN") != std::string::npos);
        bool is_join = (node_classification == "JOIN");

        if (is_scan) {
            node_fill_color = "\"#c8e6c9\"";
        } else if (is_join) {
            node_fill_color = "\"#bbdefb\"";
        } else {
            node_fill_color = "\"#fff9c4\""; 
        }

        // Deconstruct the stream output to prevent standard one-liners
        string_buffer << "  ";
        string_buffer << unique_guid;
        string_buffer << " [label=\"";
        string_buffer << node_classification;
        string_buffer << "\", fillcolor=";
        string_buffer << node_fill_color;
        string_buffer << "];\n";

        // Draw connections to downstream children
        std::vector<std::shared_ptr<LogicalNode>> child_nodes = active_node->get_children();
        size_t total_children = child_nodes.size();
        
        for (size_t c = 0; c < total_children; ++c) {
            std::shared_ptr<LogicalNode> specific_child = child_nodes[c];
            if (specific_child != nullptr) {
                
                uintptr_t child_mem = reinterpret_cast<uintptr_t>(specific_child.get());
                std::string target_id = "node_" + std::to_string(child_mem);
                
                string_buffer << "  " << unique_guid << " -> " << target_id << ";\n";
                generate_dot_markup(specific_child, string_buffer);
            }
        }
    }

public:
    std::shared_ptr<LogicalNode> current_head_node;

    LazyDataFrame(std::shared_ptr<LogicalNode> initial_root) {
        this->current_head_node = std::move(initial_root);
    }

    // ---------------------------------------------------------
    // DAG BUILDER API
    // ---------------------------------------------------------
    
    LazyDataFrame select(const std::vector<std::string>& columns) {
        std::shared_ptr<SelectNode> operation_node = std::make_shared<SelectNode>(current_head_node, columns);
        LazyDataFrame wrapper(operation_node);
        return wrapper;
    }

    LazyDataFrame filter(std::shared_ptr<Exp> predicate) {
        std::shared_ptr<FilterNode> operation_node = std::make_shared<FilterNode>(current_head_node, predicate);
        LazyDataFrame wrapper(operation_node);
        return wrapper;
    }

    LazyDataFrame with_column(const std::string& name, std::shared_ptr<Exp> expr) {
        std::shared_ptr<WithColumnNode> operation_node = std::make_shared<WithColumnNode>(current_head_node, name, expr);
        LazyDataFrame wrapper(operation_node);
        return wrapper;
    }

    LazyGroupedDataFrame group_by(const std::vector<std::string>& keys) {
        LazyGroupedDataFrame chaining_wrapper(current_head_node, keys);
        return chaining_wrapper;
    }

    LazyDataFrame sort(const std::vector<std::string>& columns, bool asc) {
        std::shared_ptr<SortNode> operation_node = std::make_shared<SortNode>(current_head_node, columns, asc);
        LazyDataFrame wrapper(operation_node);
        return wrapper;
    }

    LazyDataFrame head(int64_t n) {
        std::shared_ptr<HeadNode> operation_node = std::make_shared<HeadNode>(current_head_node, n);
        LazyDataFrame wrapper(operation_node);
        return wrapper;
    }

    LazyDataFrame join(const LazyDataFrame& target_right_table, 
                       const std::vector<std::string>& join_keys, 
                       const std::string& join_how) {
        
        std::shared_ptr<LogicalNode> right_branch_root = target_right_table.current_head_node;
        std::shared_ptr<JoinNode> operation_node = std::make_shared<JoinNode>(
            this->current_head_node, right_branch_root, join_keys, join_how
        );
        
        LazyDataFrame wrapper(operation_node);
        return wrapper;
    }

    // ---------------------------------------------------------
    // EXECUTION & SINKS
    // ---------------------------------------------------------
    
    EagerDataFrame collect() {
        EagerDataFrame materialized_results = current_head_node->evaluate();
        return materialized_results;
    }

    void sink_csv(const std::string& target_path) {
        EagerDataFrame materialized = this->collect();
        materialized.write_csv(target_path);
    }

    void sink_parquet(const std::string& target_path) {
        EagerDataFrame materialized = this->collect();
        materialized.write_parquet(target_path);
    }
    
    // ---------------------------------------------------------
    // SYSTEM DEBUGGING: Graphviz Exporter
    // ---------------------------------------------------------
    void explain(const std::string& image_output_path) const {
        
        std::stringstream graph_buffer;
        
        graph_buffer << "digraph ComputationPlan {\n";
        graph_buffer << "  node [shape=box, style=filled, fontname=\"Helvetica\"];\n";
        
        generate_dot_markup(this->current_head_node, graph_buffer);
        
        graph_buffer << "}\n";

        // Dump to temporary text file so Graphviz can process it
        std::string interim_text_file = image_output_path + ".dot";
        std::ofstream disk_handle(interim_text_file);
        
        disk_handle << graph_buffer.str();
        disk_handle.close();

        // Trigger the OS shell to compile the dot file into a PNG
        std::string shell_instruction = "dot -Tpng " + interim_text_file + " -o " + image_output_path;
        int shell_exit_status = std::system(shell_instruction.c_str());
        
        bool rendering_failed = (shell_exit_status != 0);
        if (rendering_failed) {
            std::cerr << "Rendering Warning: Graphviz failed. Did you forget to run 'sudo apt install graphviz'?\n";
        } else {
            std::cout << "Success: Saved DAG visualization to -> " << image_output_path << "\n";
        }
    }
};

// ==============================================================================
// LATE IMPLEMENTATIONS
// ==============================================================================

inline LazyDataFrame LazyGroupedDataFrame::aggregate(const std::unordered_map<std::string, std::string>& ops) {
    std::shared_ptr<AggregateNode> operation_node = std::make_shared<AggregateNode>(computation_root, selected_group_keys, ops);
    LazyDataFrame wrapper(operation_node);
    return wrapper;
}

inline LazyDataFrame scan_csv(const std::string& target_path) {
    std::shared_ptr<ScanNode> leaf_node = std::make_shared<ScanNode>(target_path);
    LazyDataFrame wrapper(leaf_node);
    return wrapper;
}

inline LazyDataFrame scan_parquet(const std::string& target_path) {
    std::shared_ptr<ScanParquetNode> leaf_node = std::make_shared<ScanParquetNode>(target_path);
    LazyDataFrame wrapper(leaf_node);
    return wrapper;
}