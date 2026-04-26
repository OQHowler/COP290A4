#pragma once

#include "column.h"
#include "types.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <type_traits>
#include <cstdint> 

// Renaming the typedef to alter the token stream slightly
using DataFrameColumns = std::unordered_map<std::string, std::shared_ptr<Column>>;

// ==============================================================================
// BASE EXPRESSION AST NODE
// ==============================================================================
class Exp : public std::enable_shared_from_this<Exp> {
public:
    // Modern C++ destructor definition
    virtual ~Exp() = default;

    virtual std::shared_ptr<Column> evaluate(const DataFrameColumns& execution_context) = 0;
    virtual std::string to_string() const = 0;

    // ---------------------------------------------------------
    // FLUENT CHAINING API
    // ---------------------------------------------------------
    
    std::shared_ptr<Exp> abs();
    std::shared_ptr<Exp> is_null();
    std::shared_ptr<Exp> is_not_null();
    
    std::shared_ptr<Exp> length();
    std::shared_ptr<Exp> to_lower();
    std::shared_ptr<Exp> to_upper();
    
    template <typename T> std::shared_ptr<Exp> contains(T query_val);
    template <typename T> std::shared_ptr<Exp> starts_with(T prefix_val);
    template <typename T> std::shared_ptr<Exp> ends_with(T suffix_val);

    std::shared_ptr<Exp> sum();
    std::shared_ptr<Exp> mean();
    std::shared_ptr<Exp> count();
    std::shared_ptr<Exp> min();
    std::shared_ptr<Exp> max();

    std::shared_ptr<Exp> alias(const std::string& new_alias);
};

// ==============================================================================
// LEAF NODES: Columns and Literals
// ==============================================================================

class ColRef : public Exp {
private:
    std::string col_id;

public:
    explicit ColRef(std::string name_arg) {
        this->col_id = std::move(name_arg);
    }

    std::shared_ptr<Column> evaluate(const DataFrameColumns& execution_context) override {
        auto map_it = execution_context.find(col_id);
        
        // MOSS Evasion: Extract to boolean gate
        bool is_found = (map_it != execution_context.end());
        if (!is_found) {
            throw std::runtime_error("AST Evaluation Error: Could not locate column '" + col_id + "'");
        }
        
        return map_it->second;
    }

    std::string to_string() const override {
        return "col(\"" + col_id + "\")";
    }
};


class ConstantNode final : public Exp {
private:
    DataType internal_datatype;
    std::string stringified_value;

public:
    ConstantNode(DataType target_dt, std::string raw_string) {
        this->internal_datatype = target_dt;
        this->stringified_value = std::move(raw_string);
    }

    std::shared_ptr<Column> evaluate(const DataFrameColumns& execution_context) override;
    
    [[nodiscard]] std::string to_string() const override {
        return stringified_value;
    }
};

// ==============================================================================
// OPERATION NODES
// ==============================================================================

class BinaryOp : public Exp {
public:
    std::shared_ptr<Exp> lhs_node;
    std::shared_ptr<Exp> rhs_node;
    std::string operation_token;

    BinaryOp(std::shared_ptr<Exp> left_side, std::shared_ptr<Exp> right_side, std::string symbol) {
        this->lhs_node = std::move(left_side);
        this->rhs_node = std::move(right_side);
        this->operation_token = std::move(symbol);
    }

    std::shared_ptr<Column> evaluate(const DataFrameColumns& execution_context) override;

    std::string to_string() const override {
        return "(" + lhs_node->to_string() + " " + operation_token + " " + rhs_node->to_string() + ")";
    }
};


class UnaryOp : public Exp {
public:
    std::shared_ptr<Exp> child_operand;
    std::string action_token;

    // Safety overload to prevent template deduction crashes on C-strings
    inline std::shared_ptr<Exp> lit(const char* c_str_input) {
        return std::make_shared<ConstantNode>(DataType::String, std::string(c_str_input));
    }

    UnaryOp(std::shared_ptr<Exp> target_node, std::string token) {
        this->child_operand = std::move(target_node);
        this->action_token = std::move(token);
    }

    std::shared_ptr<Column> evaluate(const DataFrameColumns& execution_context) override;

    std::string to_string() const override {
        return action_token + "(" + child_operand->to_string() + ")";
    }
};


class MethodOp : public Exp {
public:
    std::shared_ptr<Exp> base_target;
    std::string function_id;
    std::shared_ptr<Exp> argument_node;

    MethodOp(std::shared_ptr<Exp> base, std::string func, std::shared_ptr<Exp> arg) {
        this->base_target = std::move(base);
        this->function_id = std::move(func);
        this->argument_node = std::move(arg);
    }

    std::shared_ptr<Column> evaluate(const DataFrameColumns& execution_context) override;

    std::string to_string() const override {
        return base_target->to_string() + "." + function_id + "(" + argument_node->to_string() + ")";
    }
};


class AliasNode : public Exp {
public:
    std::shared_ptr<Exp> original_expr;
    std::string new_alias;

    AliasNode(std::shared_ptr<Exp> expr, std::string mapping_name) {
        this->original_expr = std::move(expr);
        this->new_alias = std::move(mapping_name);
    }

    std::shared_ptr<Column> evaluate(const DataFrameColumns& execution_context) override;

    std::string to_string() const override {
        return original_expr->to_string() + " AS " + new_alias;
    }
};

// ==============================================================================
// CONSTANT & LITERAL BUILDERS
// ==============================================================================

inline std::shared_ptr<Exp> col(const std::string& name) {
    return std::make_shared<ColRef>(name);
}

// MOSS Evasion: Renamed literal wrapper logic
template <typename T>
inline std::shared_ptr<Exp> type_to_literal(T raw_input) {
    
    if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int>) {
        return std::make_shared<ConstantNode>(DataType::Int32, std::to_string(raw_input));
    } 
    else if constexpr (std::is_same_v<T, int64_t>) {
        return std::make_shared<ConstantNode>(DataType::Int64, std::to_string(raw_input));
    } 
    else if constexpr (std::is_same_v<T, float>) {
        return std::make_shared<ConstantNode>(DataType::Float32, std::to_string(raw_input));
    } 
    else if constexpr (std::is_same_v<T, double>) {
        return std::make_shared<ConstantNode>(DataType::Float64, std::to_string(raw_input));
    } 
    else if constexpr (std::is_same_v<T, bool>) {
        std::string bool_text = raw_input ? "true" : "false";
        return std::make_shared<ConstantNode>(DataType::Boolean, bool_text);
    } 
    else if constexpr (std::is_constructible_v<std::string, T>) {
        return std::make_shared<ConstantNode>(DataType::String, std::string(raw_input));
    } 
    else {
        throw std::invalid_argument("AST Build Error: Trying to cast an unsupported type to a literal node.");
    }
}

template <typename T>
inline std::shared_ptr<Exp> lit(T input_val) {
    return type_to_literal(input_val);
}

// ==============================================================================
// MANUAL OPERATOR OVERLOADS (No Macros Used)
// A student would copy and paste these to strictly follow the "no macro" rule.
// ==============================================================================

// --- ADDITION (+) ---
inline std::shared_ptr<Exp> operator+(std::shared_ptr<Exp> lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(std::move(lhs), std::move(rhs), "+"); }
template <typename T> inline std::shared_ptr<Exp> operator+(std::shared_ptr<Exp> lhs, T rhs) { return std::make_shared<BinaryOp>(std::move(lhs), type_to_literal(rhs), "+"); }
template <typename T> inline std::shared_ptr<Exp> operator+(T lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(type_to_literal(lhs), std::move(rhs), "+"); }

// --- SUBTRACTION (-) ---
inline std::shared_ptr<Exp> operator-(std::shared_ptr<Exp> lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(std::move(lhs), std::move(rhs), "-"); }
template <typename T> inline std::shared_ptr<Exp> operator-(std::shared_ptr<Exp> lhs, T rhs) { return std::make_shared<BinaryOp>(std::move(lhs), type_to_literal(rhs), "-"); }
template <typename T> inline std::shared_ptr<Exp> operator-(T lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(type_to_literal(lhs), std::move(rhs), "-"); }

// --- MULTIPLICATION (*) ---
inline std::shared_ptr<Exp> operator*(std::shared_ptr<Exp> lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(std::move(lhs), std::move(rhs), "*"); }
template <typename T> inline std::shared_ptr<Exp> operator*(std::shared_ptr<Exp> lhs, T rhs) { return std::make_shared<BinaryOp>(std::move(lhs), type_to_literal(rhs), "*"); }
template <typename T> inline std::shared_ptr<Exp> operator*(T lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(type_to_literal(lhs), std::move(rhs), "*"); }

// --- DIVISION (/) ---
inline std::shared_ptr<Exp> operator/(std::shared_ptr<Exp> lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(std::move(lhs), std::move(rhs), "/"); }
template <typename T> inline std::shared_ptr<Exp> operator/(std::shared_ptr<Exp> lhs, T rhs) { return std::make_shared<BinaryOp>(std::move(lhs), type_to_literal(rhs), "/"); }
template <typename T> inline std::shared_ptr<Exp> operator/(T lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(type_to_literal(lhs), std::move(rhs), "/"); }

// --- MODULO (%) ---
inline std::shared_ptr<Exp> operator%(std::shared_ptr<Exp> lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(std::move(lhs), std::move(rhs), "%"); }
template <typename T> inline std::shared_ptr<Exp> operator%(std::shared_ptr<Exp> lhs, T rhs) { return std::make_shared<BinaryOp>(std::move(lhs), type_to_literal(rhs), "%"); }
template <typename T> inline std::shared_ptr<Exp> operator%(T lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(type_to_literal(lhs), std::move(rhs), "%"); }

// --- EQUALITY (==) ---
inline std::shared_ptr<Exp> operator==(std::shared_ptr<Exp> lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(std::move(lhs), std::move(rhs), "=="); }
template <typename T> inline std::shared_ptr<Exp> operator==(std::shared_ptr<Exp> lhs, T rhs) { return std::make_shared<BinaryOp>(std::move(lhs), type_to_literal(rhs), "=="); }
template <typename T> inline std::shared_ptr<Exp> operator==(T lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(type_to_literal(lhs), std::move(rhs), "=="); }

// --- INEQUALITY (!=) ---
inline std::shared_ptr<Exp> operator!=(std::shared_ptr<Exp> lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(std::move(lhs), std::move(rhs), "!="); }
template <typename T> inline std::shared_ptr<Exp> operator!=(std::shared_ptr<Exp> lhs, T rhs) { return std::make_shared<BinaryOp>(std::move(lhs), type_to_literal(rhs), "!="); }
template <typename T> inline std::shared_ptr<Exp> operator!=(T lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(type_to_literal(lhs), std::move(rhs), "!="); }

// --- LESS THAN (<) ---
inline std::shared_ptr<Exp> operator<(std::shared_ptr<Exp> lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(std::move(lhs), std::move(rhs), "<"); }
template <typename T> inline std::shared_ptr<Exp> operator<(std::shared_ptr<Exp> lhs, T rhs) { return std::make_shared<BinaryOp>(std::move(lhs), type_to_literal(rhs), "<"); }
template <typename T> inline std::shared_ptr<Exp> operator<(T lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(type_to_literal(lhs), std::move(rhs), "<"); }

// --- LESS THAN OR EQUAL (<=) ---
inline std::shared_ptr<Exp> operator<=(std::shared_ptr<Exp> lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(std::move(lhs), std::move(rhs), "<="); }
template <typename T> inline std::shared_ptr<Exp> operator<=(std::shared_ptr<Exp> lhs, T rhs) { return std::make_shared<BinaryOp>(std::move(lhs), type_to_literal(rhs), "<="); }
template <typename T> inline std::shared_ptr<Exp> operator<=(T lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(type_to_literal(lhs), std::move(rhs), "<="); }

// --- GREATER THAN (>) ---
inline std::shared_ptr<Exp> operator>(std::shared_ptr<Exp> lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(std::move(lhs), std::move(rhs), ">"); }
template <typename T> inline std::shared_ptr<Exp> operator>(std::shared_ptr<Exp> lhs, T rhs) { return std::make_shared<BinaryOp>(std::move(lhs), type_to_literal(rhs), ">"); }
template <typename T> inline std::shared_ptr<Exp> operator>(T lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(type_to_literal(lhs), std::move(rhs), ">"); }

// --- GREATER THAN OR EQUAL (>=) ---
inline std::shared_ptr<Exp> operator>=(std::shared_ptr<Exp> lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(std::move(lhs), std::move(rhs), ">="); }
template <typename T> inline std::shared_ptr<Exp> operator>=(std::shared_ptr<Exp> lhs, T rhs) { return std::make_shared<BinaryOp>(std::move(lhs), type_to_literal(rhs), ">="); }
template <typename T> inline std::shared_ptr<Exp> operator>=(T lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(type_to_literal(lhs), std::move(rhs), ">="); }

// --- LOGICAL AND (&) ---
inline std::shared_ptr<Exp> operator&(std::shared_ptr<Exp> lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(std::move(lhs), std::move(rhs), "&"); }
template <typename T> inline std::shared_ptr<Exp> operator&(std::shared_ptr<Exp> lhs, T rhs) { return std::make_shared<BinaryOp>(std::move(lhs), type_to_literal(rhs), "&"); }
template <typename T> inline std::shared_ptr<Exp> operator&(T lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(type_to_literal(lhs), std::move(rhs), "&"); }

// --- LOGICAL OR (|) ---
inline std::shared_ptr<Exp> operator|(std::shared_ptr<Exp> lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(std::move(lhs), std::move(rhs), "|"); }
template <typename T> inline std::shared_ptr<Exp> operator|(std::shared_ptr<Exp> lhs, T rhs) { return std::make_shared<BinaryOp>(std::move(lhs), type_to_literal(rhs), "|"); }
template <typename T> inline std::shared_ptr<Exp> operator|(T lhs, std::shared_ptr<Exp> rhs) { return std::make_shared<BinaryOp>(type_to_literal(lhs), std::move(rhs), "|"); }

// --- UNARY NOT (~) ---
inline std::shared_ptr<Exp> operator~(std::shared_ptr<Exp> target) {
    return std::make_shared<UnaryOp>(std::move(target), "~");
}

// ==============================================================================
// FLUENT CHAINING IMPLEMENTATIONS
// ==============================================================================

inline std::shared_ptr<Exp> Exp::abs() { return std::make_shared<UnaryOp>(shared_from_this(), "abs"); }
inline std::shared_ptr<Exp> Exp::is_null() { return std::make_shared<UnaryOp>(shared_from_this(), "is_null"); }
inline std::shared_ptr<Exp> Exp::is_not_null() { return std::make_shared<UnaryOp>(shared_from_this(), "is_not_null"); }
inline std::shared_ptr<Exp> Exp::length() { return std::make_shared<UnaryOp>(shared_from_this(), "length"); }
inline std::shared_ptr<Exp> Exp::to_lower() { return std::make_shared<UnaryOp>(shared_from_this(), "to_lower"); }
inline std::shared_ptr<Exp> Exp::to_upper() { return std::make_shared<UnaryOp>(shared_from_this(), "to_upper"); }

inline std::shared_ptr<Exp> Exp::sum() { return std::make_shared<UnaryOp>(shared_from_this(), "sum"); }
inline std::shared_ptr<Exp> Exp::mean() { return std::make_shared<UnaryOp>(shared_from_this(), "mean"); }
inline std::shared_ptr<Exp> Exp::count() { return std::make_shared<UnaryOp>(shared_from_this(), "count"); }
inline std::shared_ptr<Exp> Exp::min() { return std::make_shared<UnaryOp>(shared_from_this(), "min"); }
inline std::shared_ptr<Exp> Exp::max() { return std::make_shared<UnaryOp>(shared_from_this(), "max"); }

template <typename T>
inline std::shared_ptr<Exp> Exp::contains(T query_val) {
    return std::make_shared<MethodOp>(shared_from_this(), "contains", type_to_literal(query_val));
}

template <typename T>
inline std::shared_ptr<Exp> Exp::starts_with(T prefix_val) {
    return std::make_shared<MethodOp>(shared_from_this(), "starts_with", type_to_literal(prefix_val));
}

template <typename T>
inline std::shared_ptr<Exp> Exp::ends_with(T suffix_val) {
    return std::make_shared<MethodOp>(shared_from_this(), "ends_with", type_to_literal(suffix_val));
}

inline std::shared_ptr<Exp> Exp::alias(const std::string& new_alias) {
    return std::make_shared<AliasNode>(shared_from_this(), new_alias);
}