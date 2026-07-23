# Architecture Proposal: Dimension-Decoupled Presburger Tensor Verification (RFC-005)

**Document Status:** Technical RFC / Proposal  
**Target Subsystems:** `lirien-ir` (Refinement & Type Builder), `lirien-verify` (Z3 SMT Backend), `lirien/stdlib` (Stdlib Annotations)  
**Author:** Collaborative Compiler Design  

---

## 1. Executive Summary & Problem Statement

### The Problem
Currently, when verifying multi-dimensional Tensor operations (`Tensor[f32, M, N]`, 2D pooling/convolutions, vector norms, and elementwise/reduction ops), `lirien-verify` translates multi-dimensional index expressions like `tensor[i, j]` into fully linearized 1D byte offset arithmetic:
$$\text{offset} = (i \cdot S_0 + j \cdot S_1 + k) \cdot \text{element\_bytes}$$

Passing non-linear byte offset multiplication formulas (`imul.i64`) into Z3 forces the solver into general non-linear integer and floating-point theory solving (`QF_NIA` / `QF_ABV`). This causes severe SMT solver performance degradation:
1. **Non-Linear Multiplication Bottlenecks:** Z3's SMT solver cannot efficiently solve non-linear symbolic variable multiplications over cyclic loop back-edges within bounded time.
2. **SMT Solver Timeouts:** Functions with dynamic 4D spatial loops (`max_pool2d`, `convolve2d_padded`) or floating-point transcendental reductions (`l2_normalize`, `rms_norm`) fail SMT verification with `Z3 returned Unknown: canceled`.
3. **Forced Fallbacks to `@jit`:** Standard library operations were forced to rely on dynamic `@jit` runtime checks instead of compile-time static formal verification (`@verify`).

### The Proposed Solution
We propose **Dimension-Decoupled Presburger Verification (RFC-005)**.

By recognizing that multi-dimensional row-major memory buffers satisfy $0 \le \text{offset} < \text{capacity\_in\_bytes}$ **if and only if** every individual dimension index $d_k$ satisfies $0 \le d_k < \text{Dim}_k$, we can decouple the SMT verification problem:
1. **Decouple Bounds Check from Linearized Offsets:** `lirien-verify` proves $D$ independent linear Presburger inequalities ($0 \le i < \text{Dim}_0 \land 0 \le j < \text{Dim}_1 \dots$) using Z3's deterministic `QF_LIA` (Linear Integer Arithmetic) solver.
2. **Isolate Transcendental Floating-Point Arithmetic:** Mathematical computations (`sqrt`, `exp`, `log`) are decoupled from memory bounds verification. Z3 verifies memory safety in $O(1)$ linear time, leaving floating-point values unconstrained during memory safety proofs.
3. **Full `@verify` Promotion:** Upgrades all stdlib operations (including generic 4D spatial pooling, convolutions, and layer/RMS normalization) from `@jit` to **`@verify`** with $O(1)$ instant verification times (< 5ms).

---

## 2. Architectural Pipeline

```
[Python AST & Tensor Refinement Types]
                 │
                 ▼
[lirien-ir: Dimension Decomposer]
  ├── Extracts rank D and dimension shape bounds (Dim_0, Dim_1, ..., Dim_D)
  └── Maps index tuple (i_0, i_1, ..., i_D) -> [Decoupled 1D Bounds Predicates]
                 │
                 ▼
[lirien-verify: Presburger Linear SMT Solver]
  ├── Proves: 0 <= i_k < Dim_k for all k in [0, D)  <── (Linear QF_LIA -> O(D) Time)
  └── Bypasses non-linear multiplication (i_0 * S_0 + i_1 * S_1)
                 │  (Proof Successful)
                 ▼
[lirien-backend / Cranelift]
  └── Lowers linearized offset address math into native machine instructions
```

---

## 3. Mathematical Foundations of Dimension Decoupling

For a row-major tensor $A \in \mathbb{R}^{\text{Dim}_0 \times \text{Dim}_1 \times \dots \times \text{Dim}_{D-1}}$ with strides $S_k = \prod_{m=k+1}^{D-1} \text{Dim}_m$:

### Theorem (Dimension Decoupled Bounds Theorem)
If $0 \le i_k < \text{Dim}_k$ for all $k \in \{0, 1, \dots, D-1\}$, then:
$$0 \le \sum_{k=0}^{D-1} i_k S_k < \prod_{k=0}^{D-1} \text{Dim}_k$$

### SMT Complexity Shift:
- **Linearized Form (Old):** Z3 solves $\text{Assert}\left(0 \le \sum_{k=0}^{D-1} i_k \cdot S_k \cdot 4 < \text{TotalBytes}\right) \implies \text{Non-Linear Integer Logic (Exponential Time)}$.
- **Decoupled Form (RFC-005):** Z3 solves $\bigwedge_{k=0}^{D-1} \text{Assert}(0 \le i_k < \text{Dim}_k) \implies \text{Presburger Linear Integer Logic (Linear Time } O(D)\text{)}$.

---

## 4. Implementation Specifications

### 4.1 `lirien-ir` (Type & Index Metadata Extension)
- Extend `InstructionKind::TensorLoad` and `InstructionKind::TensorStore` to attach dimension bound metadata:
  ```rust
  pub struct TensorAccessMeta {
      pub indices: Vec<Value>,
      pub dimensions: Vec<Value>,
      pub element_size: usize,
  }
  ```

### 4.2 `lirien-verify` (Presburger Rule Addition)
- Update `translate_tensor_load` and `translate_tensor_store` in `crates/lirien-verify/src/verifier/memory.rs`:
  Instead of asserting `bv_sge(linearized_offset, 0)` and `bv_slt(linearized_offset, capacity)`, emit $D$ Presburger assertions:
  ```rust
  for (idx_val, dim_val) in meta.indices.iter().zip(meta.dimensions.iter()) {
      let idx_z3 = ctx.z3_bvs.get(idx_val).unwrap();
      let dim_z3 = ctx.z3_bvs.get(dim_val).unwrap();
      
      let ge_zero = ctx.backend.bv_sge(idx_z3, &zero_bv);
      let lt_dim = ctx.backend.bv_slt(idx_z3, dim_z3);
      let in_bounds = ctx.backend.bool_and(&[ge_zero, lt_dim]);
      
      ctx.safety_checks.push(SafetyCheck {
          path_cond: path_cond.clone(),
          violation_cond: ctx.backend.bool_not(&in_bounds),
          error_message: format!("Tensor dimension index out of bounds"),
          location: inst.location,
      });
  }
  ```

---

## 5. Promotion Plan for `num` Standard Library

Once RFC-005 is implemented, the following functions in `num` will be upgraded from `@jit` to **`@verify`**:
1. `max_pool2d` (`num.nn`)
2. `avg_pool2d` (`num.nn`)
3. `convolve2d_padded` (`num.nn`)
4. `resize_nearest` (`num.nn`)
5. `l2_normalize` (`num.nn`)
6. `rms_norm` (`num.nn`)
7. `layer_norm` (`num.nn`)
8. `hardsigmoid` (`num.activations`)
9. `hardswish` (`num.activations`)
10. `sigmoid_cross_entropy` (`num.training`)
11. `rms_norm_simd` (`num.simd`)
12. `layer_norm_simd` (`num.simd`)
