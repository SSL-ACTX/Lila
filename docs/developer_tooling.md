# Developer Tooling & Diagnostics

Lirien includes several tooling APIs and diagnostic aids to help you inspect compiler output, debug verification issues, and optimize hot code paths.

---

## Bypassing Verification (`@jit`)

The `@jit` decorator bypasses the Z3 formal verification stage entirely, compiling your function directly to native machine code using Cranelift. 

This is useful for:
*   Functions whose correctness is already checked upstream.
*   Functions with non-verifiable constructs (e.g. interfacing with external unsafe C libraries).
*   Speeding up development compilation times for trusted helpers.

```python
from lirien import jit, i64


@jit
def fast_add(a: i64, b: i64) -> i64:
    # Compiles directly via Cranelift, bypassing Z3 verification
    return a + b
```

---

## Direct Inline Cranelift Blocks (`with clif(...)`)

For high-performance low-level kernel optimization and direct register-level memory operations, Lirien supports inline Cranelift IR blocks using the `with clif(...)` context manager.

This allows you to map Python variables directly to SSA virtual registers (`v0`, `v1`, `v2`, `v3`), execute direct arithmetic and pointer offset loads/stores (`v0[offset]`), and bind output values back to Python variables.

```python
from lirien import verify, i64, Box
from lirien.clif import clif, v0, v1, v2, v3


@verify
def clif_kernel(a: i64, b: i64) -> i64:
    offset = 42
    with clif(inputs={a: v0, b: v1, offset: v2}, outputs={v3: "res"}):
        v4 = v0 + v2
        v3 = v4 // v1
    return res


@verify
def clif_store(ptr: Box[i64], val: i64) -> None:
    with clif(inputs={ptr: v0, val: v1}):
        v0[0] = val  # Direct pointer store at offset 0
```

---

## Scoped Verification Bypass (`no_verification`)

If you want to compile and run tests or run benchmarks without the overhead of Z3 compilation latency, you can run your imports or definitions inside the `no_verification()` context manager. This disables solver checks globally for all `@verify` functions defined within the block.

```python
from lirien import verify, no_verification, i64

with no_verification():
    # These functions compile immediately via Cranelift only
    @verify
    def complex_math(x: i64) -> i64:
        return x * x
```

---

## Hierarchical Compiler Tracing (`tracing()`)

Lirien uses the Rust `tracing` crate under the hood to output logging information. The `tracing()` context manager allows you to configure target log levels dynamically for specific compilation subsystems for the lifetime of a block.

```python
from lirien import verify, tracing, LIVENESS, VERIFY, SSA, Z3

# Enable trace levels for Z3 queries, info for SSA, and debug for verify
with tracing({VERIFY: "debug", Z3: "trace", SSA: "info"}):

    @verify
    def target(x: i64) -> i64:
        return x + 1
```

Available subsystems:
*   `VERIFY`: Safety verification logic and general JIT compiler diagnostics.
*   `Z3`: SMT solver constraint generation and solver stats.
*   `SSA`: Control flow graph extraction, builder activity, and type inference.
*   `LIVENESS`: Variable lifetime analysis.
*   `BACKEND`: Cranelift lowering, optimization, and code generation.

For persistent global logging, use `configure_tracing()`:
```python
from lirien import configure_tracing, BACKEND

configure_tracing({BACKEND: "warn"})
```

---

## Source-Level Error Locations

When a Z3 check fails, Lirien maps the failure back to your Python source file, displaying the line, column, and code context.

Example error message:
```text
[Lirien Warning] Lirien Verification Failed for 'divide_unsafe': Potential division by zero at v2
  --> source.py:3:12
   |
 3 |    return n // d
   |                ^--- Logic error detected here
```

Every Lirien IR instruction holds `SourceLocation` metadata carrying the absolute file path, line, and column indices, making debugging verification and code lowering simple.
