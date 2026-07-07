# Experimental `num` Standard Library

Lirien includes an experimental standard library package for high-performance numerical computing and neural networks under `lirien.stdlib.num`. 

This library provides JIT-compiled, SIMD-accelerated, and Z3-verified operations for tensor arithmetic, neural network layers, activations, loss functions, and optimization steps.

---

## Core Design Principles

1.  **Shape Safety at Type-Level**: The library defines standard type-level dimensions (`B`, `M`, `N`, `K`, `H`, `W`, etc.) using `TypeVar`. Functions enforce exact shape compatibility (e.g., matching inner dimensions for matrix multiplication) statically.
2.  **Provable Safety**: Every operator in `lirien.stdlib.num` is decorated with `@verify`. Z3 formally proves the absence of out-of-bounds memory accesses and division-by-zero errors at compile time.
3.  **SIMD Acceleration**: By utilizing Lirien's native 128-bit vector types (such as `f32x4`), the operators map directly to hardware vector registers, bypassing Python loop overhead and auto-vectorization uncertainty.

---

## API Categories

### 1. Dimension TypeVars (`lirien.stdlib.num.shared`)
Pre-declared type variables representing dimensions:
*   `B`: Batch size
*   `M`, `N`, `K`: General matrix and tensor dimensions
*   `H`, `W`: Height and Width (for images/features)
*   `KH`, `KW`: Kernel Height and Kernel Width (for convolutions)
*   `OH`, `OW`: Output Height and Output Width
*   `SH`, `SW`: Stride Height and Stride Width
*   `PH`, `PW`: Padding Height and Padding Width

### 2. Basic Tensor Operations (`lirien.stdlib.num.ops`)
Standard numerical operations:
*   `transpose(a: Tensor[f32, M, N], out: Tensor[f32, N, M])`
*   `matmul(a: Tensor[f32, M, N], b: Tensor[f32, N, K], out: Tensor[f32, M, K])`
*   `add` / `sub` / `mul(a: Tensor[f32, M, N], b: Tensor[f32, M, N], out: Tensor[f32, M, N])`
*   `clip(a: Tensor[f32, M, N], out: Tensor[f32, M, N], min_val: f32, max_val: f32)`
*   `mean(a: Tensor[f32, M], out: Tensor[f32, 1], n: f32)` (requires precondition `n > 0.0`)
*   `scale(a: Tensor[f32, M, N], out: Tensor[f32, M, N], factor: f32)`
*   `bias_add(a: Tensor[f32, M, N], bias: Tensor[f32, N], out: Tensor[f32, M, N])`
*   `matvec(matrix: Tensor[f32, M, N], vector: Tensor[f32, N], out: Tensor[f32, M])`
*   `outer(a: Tensor[f32, M], b: Tensor[f32, N], out: Tensor[f32, M, N])`
*   `dot(a: Tensor[f32, M], b: Tensor[f32, M], out: Tensor[f32, 1])`
*   `bmm(a: Tensor[f32, B, M, N], b: Tensor[f32, B, N, K], out: Tensor[f32, B, M, K])` (Batched Matrix Multiplication)

### 3. Activations (`lirien.stdlib.num.activations`)
Common deep learning activation functions:
*   `relu`, `leaky_relu`, `elu`, `selu`, `gelu`
*   `sigmoid`, `hardsigmoid`, `silu` (Swish), `hardswish`, `swiglu`
*   `softmax(a: Tensor[f32, M, N], out: Tensor[f32, M, N])`

### 4. Neural Network Layers (`lirien.stdlib.num.nn`)
Core signal processing and feature representation layers:
*   `convolve1d`, `convolve2d`, `convolve2d_padded` (cross-correlations)
*   `max_pool2d_2x2`, `avg_pool2d_2x2`, `max_pool2d`, `avg_pool2d`
*   `standardize`, `l1_normalize`, `l2_normalize`
*   `layer_norm`, `rms_norm`
*   `cosine_similarity`
*   `resize_nearest` (nearest neighbor image interpolation)

### 5. SIMD-Accelerated Operators (`lirien.stdlib.num.simd`)
Manual vectorizations using 128-bit `f32x4` lane types:
*   `dot_simd`, `matvec_simd`, `matmul_simd`, `bmm_simd`
*   `add_simd`, `sub_simd`, `mul_simd`, `scale_simd`, `div_simd`
*   `relu_simd`
*   `layer_norm_simd`, `rms_norm_simd`
*   `mse_simd` (Mean Squared Error), `mae_simd` (Mean Absolute Error)

### 6. Training & Optimizers (`lirien.stdlib.num.training`)
JIT-compiled backpropagation and optimizer routines:
*   `sgd_momentum_step(param: Tensor[f32, M], grad: Tensor[f32, M], velocity: Tensor[f32, M], lr: f32, momentum: f32)`
*   `adamw_step(...)` (AdamW parameter updates with weight decay, 1st/2nd moment tracking)
*   `sigmoid_cross_entropy`, `softmax_cross_entropy_with_logits`
*   `l2_loss`

---

## Code Examples

### Example 1: Type-Safe Matrix Multiplication
In this example, Lirien verifies that shapes match and loop bounds are memory-safe.

```python
from lirien import verify, Tensor, f32
from lirien.stdlib.num import matmul, M, N, K

# Create tensors with concrete shapes
# M=2, N=3, K=4
A = Tensor.alloc((2, 3), f32)
B = Tensor.alloc((3, 4), f32)
C = Tensor.alloc((2, 4), f32)

# Populating values...
# Execute verified, Gil-free matmul
matmul(A, B, C)
```

### Example 2: Vectorized Dot Product (`dot_simd`)
This utilizes 128-bit floats (`f32x4`) to process 4 elements per instruction cycle.

```python
from lirien import verify, Tensor, f32, f32x4
from lirien.stdlib.num import dot_simd

# Allocate arrays where each cell holds a 4x32-bit float vector
M_dim = 1000
A = Tensor.alloc((M_dim,), f32x4)
B = Tensor.alloc((M_dim,), f32x4)
Out = Tensor.alloc((1,), f32)

# Runs in SIMD lane execution
dot_simd(A, B, Out)
```
