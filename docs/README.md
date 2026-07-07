# Lirien Documentation Hub

Welcome to the official documentation for **Lirien**, a verifying JIT compiler for a safe subset of Python using Z3 and Cranelift.

Lirien treats type annotations as formal mathematical specifications, uses an SMT solver to prove their safety at compile time, and then JIT-compiles them directly to native machine code—bypassing the CPython interpreter and the GIL.

---

## Documentation Sections

Explore the different facets of Lirien:

*   **[Getting Started & Installation](getting_started.md)**: Setup prerequisites, compile the Rust core, install the Python extension, and run tests.
*   **[Core Concepts & Verification Contracts](core_concepts.md)**: Refinement types, the `V` placeholder DSL, Design by Contract (`assert`), and inductive recursive proofs.
*   **[Performance & Monomorphization](performance_dispatch.md)**: Static dispatch via `Protocol`, GIL-free parallelism, flow-sensitive type narrowing, const generics, multiple dispatch, and SIMD.
*   **[Data Structures & Memory Layouts](data_structures.md)**: Flat C-ABI structs, stack-allocated value types, ADTs, `TypedDict`, growable `List` arrays, Tensors with kernel fusion, and zero-copy buffer slicing.
*   **[Developer Tooling & Diagnostics](developer_tooling.md)**: Bypassing verification via `@jit`, the thread-local `no_verification()` block, hierarchical subsystem `tracing()`, and source-level diagnostic diagnostics.
*   **[Architecture & Compiler Pipeline](architecture_pipeline.md)**: The internal representation (SSA IR), caching pipeline, formal Z3 encoding, and Cranelift backend execution.
*   **[Experimental `num` Standard Library](stdlib_num.md)**: JIT-compiled, SIMD-accelerated, and Z3-verified operations for numerical computing, convolutions, activations, and model training.

---

## High-Level Overview

```mermaid
graph TD
    %% Python Side
    PySource["@verify<br/>def my_func(...)"]:::python

    %% Bridge / Caching
    Hash["Cache Manager<br/>(seahash)"]:::bridge
    Cache{{"Cache Hit?"}}:::bridge

    %% Frontend (AST -> IR)
    AST["AST Parser<br/>(rustpython)"]:::frontend
    Builder["SSA Builder<br/>(CFG, capture analysis)"]:::frontend
    Opt["Optimization<br/>(DCE, constant folding, type propagation)"]:::frontend

    %% Middle-end (Verification)
    Z3["Z3 SMT Solver<br/>(arithmetic, memory, control-flow)"]:::verify

    %% Backend
    CL["Cranelift JIT<br/>(machine code)"]:::backend
    Trampoline["C-ABI Trampoline<br/>(PyO3)"]:::backend

    %% Flow
    PySource --> Hash
    Hash --> Cache
    Cache -- Yes --> CL
    Cache -- No --> AST
    AST --> Builder
    Builder --> Opt
    Opt --> Z3
    Z3 --> CL
    CL --> Trampoline
    Trampoline -->|"Native call"| PySource

    classDef python fill:#306998,stroke:#FFD43B,stroke-width:2px,color:#fff;
    classDef bridge fill:#6e5494,stroke:#fff,stroke-width:1px,color:#fff;
    classDef frontend fill:#b7410e,stroke:#fff,stroke-width:1px,color:#fff;
    classDef verify fill:#0052cc,stroke:#fff,stroke-width:1px,color:#fff;
    classDef backend fill:#2c3e50,stroke:#fff,stroke-width:1px,color:#fff;
```
