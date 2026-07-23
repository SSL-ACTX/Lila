import math
from lirien import verify, f32, f32x4, Tensor
from .shared import B, M, N, K


@verify
def dot_simd(a: Tensor[f32x4, M], b: Tensor[f32x4, M], out: Tensor[f32, 1]):
    """
    SIMD-accelerated dot product of two tensors of f32x4.
    Computes parallel vector products and performs a horizontal sum.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    acc = a[0] - a[0]  # Initialize zero vector
    for i in range(M):
        acc = acc + a[i] * b[i]
    out[0] = acc[0] + acc[1] + acc[2] + acc[3]


@verify
def matvec_simd(
    matrix: Tensor[f32x4, M, N],
    vector: Tensor[f32x4, N],
    out: Tensor[f32, M],
):
    """
    SIMD-accelerated matrix-vector multiplication.
    Computes parallel row-vector dot products.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        acc = matrix[i, 0] - matrix[i, 0]  # Initialize zero vector
        for j in range(N):
            acc = acc + matrix[i, j] * vector[j]
        out[i] = acc[0] + acc[1] + acc[2] + acc[3]


@verify
def mse_simd(a: Tensor[f32x4, M], b: Tensor[f32x4, M], out: Tensor[f32, 1]):
    """
    SIMD-accelerated Mean Squared Error (MSE) accumulation.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    acc = a[0] - a[0]
    for i in range(M):
        diff = a[i] - b[i]
        acc = acc + diff * diff
    out[0] = acc[0] + acc[1] + acc[2] + acc[3]


@verify
def mae_simd(a: Tensor[f32x4, M], b: Tensor[f32x4, M], out: Tensor[f32, 1]):
    """
    SIMD-accelerated Mean Absolute Error (MAE) accumulation.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    acc = a[0] - a[0]
    for i in range(M):
        diff = a[i] - b[i]
        acc = acc + abs(diff)
    out[0] = acc[0] + acc[1] + acc[2] + acc[3]


@verify
def add_simd(
    a: Tensor[f32x4, M, N],
    b: Tensor[f32x4, M, N],
    out: Tensor[f32x4, M, N],
):
    """
    SIMD-accelerated element-wise addition of two tensors of f32x4.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] + b[i, j]


@verify
def sub_simd(
    a: Tensor[f32x4, M, N],
    b: Tensor[f32x4, M, N],
    out: Tensor[f32x4, M, N],
):
    """
    SIMD-accelerated element-wise subtraction of two tensors of f32x4.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] - b[i, j]


@verify
def mul_simd(
    a: Tensor[f32x4, M, N],
    b: Tensor[f32x4, M, N],
    out: Tensor[f32x4, M, N],
):
    """
    SIMD-accelerated element-wise multiplication of two tensors of f32x4.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] * b[i, j]


@verify
def scale_simd(
    a: Tensor[f32x4, M, N],
    out: Tensor[f32x4, M, N],
    factor: f32,
):
    """
    SIMD-accelerated element-wise scaling of a tensor of f32x4 by a scalar factor.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] * factor


@verify
def relu_simd(a: Tensor[f32x4, M, N], out: Tensor[f32x4, M, N]):
    """
    SIMD-accelerated element-wise branchless ReLU of a tensor of f32x4.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            zero = val * 0.0
            out[i, j] = max(val, zero)


@verify
def div_simd(
    a: Tensor[f32x4, M, N],
    b: Tensor[f32x4, M, N],
    out: Tensor[f32x4, M, N],
):
    """
    SIMD-accelerated element-wise division of two tensors of f32x4.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] / b[i, j]


@verify
def matmul_simd(
    a: Tensor[f32x4, M, K],
    b: Tensor[f32x4, K, N],
    out: Tensor[f32, M, N],
):
    """
    SIMD-accelerated 2D matrix multiplication.
    Computes parallel row-column vector products and performs a horizontal sum.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            acc = a[i, 0] - a[i, 0]  # Initialize zero vector
            for k in range(K):
                acc = acc + a[i, k] * b[k, j]
            out[i, j] = acc[0] + acc[1] + acc[2] + acc[3]


@verify
def bmm_simd(
    a: Tensor[f32x4, B, M, K],
    b: Tensor[f32x4, B, K, N],
    out: Tensor[f32, B, M, N],
):
    """
    SIMD-accelerated batch matrix multiplication.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for batch in range(B):
        for i in range(M):
            for j in range(N):
                acc = a[batch, i, 0] - a[batch, i, 0]  # Initialize zero vector
                for k in range(K):
                    acc = acc + a[batch, i, k] * b[batch, k, j]
                out[batch, i, j] = acc[0] + acc[1] + acc[2] + acc[3]


@verify
def rms_norm_simd(
    a: Tensor[f32x4, M],
    out: Tensor[f32x4, M],
    epsilon: f32,
    n: f32,
):
    """
    SIMD-accelerated RMSNorm.
    Requires 'epsilon > 0.0' and 'n > 0.0' (where n is float(M * 4)).
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert epsilon > 0.0
    assert n > 0.0
    acc = a[0] - a[0]  # Initialize zero vector
    for i in range(M):
        acc = acc + a[i] * a[i]
    sum_sq = acc[0] + acc[1] + acc[2] + acc[3]
    sqrt_input = abs(sum_sq / n) + epsilon
    assert sqrt_input >= 0.0
    rms = math.sqrt(sqrt_input)
    assert rms > 0.0
    inv_rms = 1.0 / rms
    for i in range(M):
        out[i] = a[i] * inv_rms


@verify
def layer_norm_simd(
    a: Tensor[f32x4, M],
    out: Tensor[f32x4, M],
    gamma: Tensor[f32x4, M],
    beta: Tensor[f32x4, M],
    epsilon: f32,
    n: f32,
):
    """
    SIMD-accelerated Layer Normalization.
    Requires 'epsilon > 0.0' and 'n > 0.0' (where n is float(M * 4)).
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert epsilon > 0.0
    assert n > 0.0

    sum_vec = a[0] - a[0]
    sum_sq_vec = a[0] - a[0]
    for i in range(M):
        val = a[i]
        sum_vec = sum_vec + val
        sum_sq_vec = sum_sq_vec + val * val

    sum_val = sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3]
    sum_sq = sum_sq_vec[0] + sum_sq_vec[1] + sum_sq_vec[2] + sum_sq_vec[3]

    mean_val = sum_val / n
    var_val = sum_sq / n - mean_val * mean_val
    sqrt_input = abs(var_val) + epsilon
    assert sqrt_input >= 0.0
    std_val = math.sqrt(sqrt_input)
    assert std_val > 0.0
    inv_std = 1.0 / std_val

    for i in range(M):
        out[i] = (a[i] - mean_val) * inv_std * gamma[i] + beta[i]
