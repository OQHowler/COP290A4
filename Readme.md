# DataFrameLib

A C++ library for tabular data processing with eager and lazy
execution modes, built on top of Apache Arrow. 

## Features

- `EagerDataFrame` — immediate, materialised execution
- `LazyDataFrame` — deferred execution backed by a computation DAG
- `QueryOptimizer` — rule-based optimisation passes applied
  automatically before plan execution
- I/O for CSV and Parquet via Apache Arrow
- Plan visualisation as PNG via Graphviz (`explain()`)
- Type-safe expression system with operator overloading

## Prerequisites

- Linux / Unix (tested on Ubuntu 22.04 and 24.04)
- CMake >= 3.16
- C++20 compiler (GCC 10+ or Clang 12+)
- Apache Arrow C++ — `libarrow-dev`
- Apache Parquet C++ — `libparquet-dev`
- Graphviz (the `dot` binary), required by `LazyDataFrame::explain()`

### Installing dependencies (Ubuntu / Debian)

    sudo apt update
    sudo apt install build-essential cmake \
                     libarrow-dev libparquet-dev graphviz


## Project Layout

    project/
    ├── CMakeLists.txt
    ├── README.md
    ├── report.pdf
    ├── include/
    │   ├── types.h               # DataType enum+promotion rules
    │   ├── column.h              # Column = Arrow array+type tag
    │   ├── exp.h                 # Expression AST
    │   ├── eager_df.h            # EagerDataFrame, GroupedDataFrame
    │   ├── lazy_df.h             # LazyDataFrame and LogicalNode tree
    │   ├── query_optimizer.h     # QueryOptimizer
    │   └── dataframelib/
    │       └── dataframelib.h    # Public umbrella header
    └── src/
        └── exp.cpp               # Expression system implementation

The CMake target name is `dataframelib`, as expected by the
autograder.

## Building

    mkdir build
    cd build
    cmake ..
    cmake --build .

This produces `libdataframelib.{a,so}` against which test programs
can link via `target_link_libraries(... dataframelib)`.

## API Reference

### DataFrame operations

| Operation                       | Eager | Lazy |
| ------------------------------- | :---: | :--: |
| `select(cols)`                  |  ✓    |  ✓   |
| `filter(expr)`                  |  ✓    |  ✓   |
| `with_column(name, expr)`       |  ✓    |  ✓   |
| `group_by(keys).aggregate(map)` |  ✓    |  ✓   |
| `join(other, keys, how)`        |  ✓    |  ✓   |
| `sort(cols, asc)`               |  ✓    |  ✓   |
| `head(n)`                       |  ✓    |  ✓   |
| `collect()`                     |  —    |  ✓   |
| `explain(path)`                 |  —    |  ✓   |
| `write_csv` / `write_parquet`   |  ✓    |  —   |
| `sink_csv`  / `sink_parquet`    |  —    |  ✓   |

Supported join modes: `inner`, `left`, `right`, `outer`.
Supported aggregations: `sum`, `mean`, `min`, `max`, `count`.

### Expression system

    col("x") + col("y")              // arithmetic (+ - * / %)
    col("x") > 30                    // comparisons
    (col("a") > 0) & (col("b") < 5)  // boolean &, |
    ~col("x").is_null()              // negation
    col("name").contains("foo")      // string ops
    col("salary").mean()             // aggregations
    col("x").alias("renamed")        // aliasing

Both `col(...) + 5` and `col(...) + lit(5)` work; primitive
operands are wrapped automatically.

### Type system and null semantics

Supported types: `Int32`, `Int64`, `Float32`, `Float64`, `String`,
`Boolean`. Type promotion follows the rule `int + float = float`.
Mixing string with numeric in any binary operation throws.
Any binary operation with at least one null operand produces null.

## Query optimisation

The optimiser runs the following passes, in this order, before
execution:

1. **Expression rewriting** (constant folding, expression
   simplification) on every `FilterNode` and `WithColumnNode`.
2. **Predicate pushdown** — filters are pushed below `select` and
   `with_column`, and split across the inputs of `join` when the
   conjuncts reference columns from only one side.
3. **Limit pushdown** — `head(n)` is pushed below `select` and
   `with_column` (never below `sort`, `filter`, `group_by`, or
   `join`, where it is not order-preserving / cardinality-safe).



## Notes on the autograder integration

- The CMake target is `dataframelib`.
- All public symbols are inside `namespace dataframelib`.
- The umbrella header `dataframelib/dataframelib.h` is the only
  header test programs need to include.
- `ARROW_THROW_NOT_OK` is defined as a polyfill in
  `dataframelib.h` for Arrow versions where the macro is no longer
  shipped (this came up on Piazza).
