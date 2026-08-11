<div align="center">

# Lirien

**A Verifying JIT Compiler for a Safe Subset of Python**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Rust](https://img.shields.io/badge/Rust-1.80+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Z3](https://img.shields.io/badge/Solver-Z3_4.12+-red.svg)](https://github.com/Z3Prover/z3)
[![Cranelift](https://img.shields.io/badge/JIT-Cranelift-purple.svg)](https://cranelift.dev/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SSL-ACTX/Lirien)

</div>

> [!NOTE]
> **Project Status: Personal Experimental Compiler**
>
> Lirien is a personal research and development project demonstrating path-aware liquid type verification for JIT-compiled Python (implementing ADTs, SIMD, monomorphization, and a verified numerical stdlib). Feature additions, RFC implementations, and maintenance are driven as an ongoing personal endeavor.

> [!WARNING]
> Lirien is an experimental research compiler. It is not production-ready and should not be used in critical systems.

---

Python's type annotations are unenforced at runtime. The standard trade-off is to accept the overhead of runtime checks and the Global Interpreter Lock (GIL) for safety, or to rewrite performance-critical code in a systems language at the cost of development complexity. 

Lirien takes a different approach: it treats type annotations as formal specifications, uses an SMT solver (Z3) to prove their correctness at compile time, and then JIT-compiles the code directly to native machine instructions (via Cranelift), bypassing the CPython interpreter and the GIL.

The result is a compiler that can statically guarantee the absence of common classes of runtime errors—division by zero, out-of-bounds array/buffer accesses, null pointer dereferences—while executing at native speed.

---

## Documentation Hub

For deep, detailed guides on individual features, check out the **[Lirien Documentation Hub](docs/README.md)**:

*   **[Getting Started & Installation](docs/getting_started.md)**: Setup prerequisites, build command, and run tests.
*   **[Core Concepts & Verification Contracts](docs/core_concepts.md)**: Refinement types, the `V` placeholder DSL, Design by Contract (`assert`), and inductive proofs.
*   **[Performance & Monomorphization](docs/performance_dispatch.md)**: Static dispatch via `Protocol`, GIL-free parallelism, flow-sensitive type narrowing, const generics, and SIMD.
*   **[Data Structures & Memory Layouts](docs/data_structures.md)**: Flat C-ABI structs, stack-allocated value types, ADTs, `TypedDict`, growable lists, and tensors with kernel fusion.
*   **[Developer Tooling & Diagnostics](docs/developer_tooling.md)**: Bypassing verification via `@jit`, context-based log `tracing()`, and source-level error messages.
*   **[Architecture & Compiler Pipeline](docs/architecture_pipeline.md)**: Internal SSA IR design, compilation steps, and the caching mechanism.
*   **[Experimental `num` Standard Library](docs/stdlib_num.md)**: JIT-compiled, SIMD-accelerated, and Z3-verified numerical operations, activations, and neural network layers.

---

## Quick Example

```python
from lirien import verify, i64, Refined, V

# A refinement type: an integer strictly greater than zero
Positive = Refined[i64, V > 0]


@verify
def divide_verified(n: i64, d: Positive) -> i64:
    # Z3 proves 'd > 0' holds. ZeroDivisionError is statically impossible.
    return n // d


print(divide_verified(100, 5))  # Executed in native JIT-compiled machine code
```

---

## Getting Started

### Quick Install
First, ensure you have Rust (1.80+), Python (3.10+), and Z3 (v4.12+) installed.

```bash
# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install maturin ruff

# Build extension module
maturin develop --release
```

### Running Tests
```bash
# Run Rust tests
cargo test

# Run Python integration tests
PYTHONPATH=./python python3 -m unittest discover tests/python
```

---

## License

Built with 🦀 & 🐍 by [Seuriin](https://github.com/SSL-ACTX). Distributed under the AGPL-3.0 License. See [LICENSE](LICENSE) for details.
