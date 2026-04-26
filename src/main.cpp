#include <iostream>
#include "lazy_df.h"
#include "query_optimizer.h"

int main() {
    std::cout << "--- Query Optimizer & Graphviz Test ---\n\n";

    // Build two basic lazy plans
    LazyDataFrame users = scan_csv("users.csv");
    LazyDataFrame salaries = scan_csv("salaries.csv");

    // Create an unoptimized DAG: Join first, THEN filter
    LazyDataFrame unoptimized_plan = users
        .join(salaries, {"id"}, "inner")
        .filter(col("age") > 30);

    // Run the optimizer
    QueryOptimizer engine;
    LazyDataFrame optimized_plan = engine.optimize(unoptimized_plan);

    std::cout << "Optimization complete! Generating execution graphs...\n";
    
    // Dump the PNGs to disk
    unoptimized_plan.explain("unoptimized_plan.png");
    optimized_plan.explain("optimized_plan.png");

    return 0;
}