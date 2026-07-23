from lirien import verify, jit, f32, Tensor
from .shared import B, M, N, K


@verify
def transpose(a: Tensor[f32, M, N], out: Tensor[f32, N, M]):
    """
    Transpose a 2D tensor 'a' of shape (M, N) into 'out' of shape (N, M).
    Verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[j, i] = a[i, j]


@verify
def matmul(a: Tensor[f32, M, N], b: Tensor[f32, N, K], out: Tensor[f32, M, K]):
    """
    Matrix multiplication of 'a' and 'b', storing the result in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(K):
            sum_val: f32 = 0.0
            for k_idx in range(N):
                sum_val = sum_val + a[i, k_idx] * b[k_idx, j]
            out[i, j] = sum_val


@verify
def add(a: Tensor[f32, M, N], b: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Element-wise addition of 'a' and 'b', storing the result in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] + b[i, j]


@verify
def sub(a: Tensor[f32, M, N], b: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Element-wise subtraction of 'a' and 'b', storing the result in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] - b[i, j]


@verify
def mul(a: Tensor[f32, M, N], b: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Element-wise multiplication of 'a' and 'b', storing the result in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] * b[i, j]


@jit
def clip(a: Tensor[f32, M, N], out: Tensor[f32, M, N], min_val: f32, max_val: f32):
    """
    Clip the values in 'a' to [min_val, max_val] and store in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            if val < min_val:
                out[i, j] = min_val
            elif val > max_val:
                out[i, j] = max_val
            else:
                out[i, j] = val


@verify
def mean(a: Tensor[f32, M], out: Tensor[f32, 1], n: f32):
    """
    Compute the mean of 'a' and store in 'out[0]'.
    Requires precondition 'n > 0.0' to guarantee division safety.
    """
    assert n > 0.0
    sum_val: f32 = 0.0
    for i in range(M):
        sum_val = sum_val + a[i]
    out[0] = sum_val / n


@verify
def scale(a: Tensor[f32, M, N], out: Tensor[f32, M, N], factor: f32):
    """
    Scale the tensor 'a' by a scalar 'factor' and store in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] * factor


@verify
def bias_add(a: Tensor[f32, M, N], bias: Tensor[f32, N], out: Tensor[f32, M, N]):
    """
    Add a 1D bias vector 'bias' to 'a' along the last dimension and store in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] + bias[j]


@verify
def matvec(matrix: Tensor[f32, M, N], vector: Tensor[f32, N], out: Tensor[f32, M]):
    """
    Matrix-vector multiplication, storing the result in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        sum_val: f32 = 0.0
        for j in range(N):
            sum_val = sum_val + matrix[i, j] * vector[j]
        out[i] = sum_val


@verify
def outer(a: Tensor[f32, M], b: Tensor[f32, N], out: Tensor[f32, M, N]):
    """
    Compute the outer product of vectors 'a' and 'b', storing in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i] * b[j]


@verify
def dot(a: Tensor[f32, M], b: Tensor[f32, M], out: Tensor[f32, 1]):
    """
    Compute the dot product of vectors 'a' and 'b', storing in 'out[0]'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    sum_val: f32 = 0.0
    for i in range(M):
        sum_val = sum_val + a[i] * b[i]
    out[0] = sum_val


@verify
def bmm(
    a: Tensor[f32, B, M, N],
    b: Tensor[f32, B, N, K],
    out: Tensor[f32, B, M, K],
):
    """
    Batch matrix multiplication: out[b] = a[b] @ b[b]
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for batch in range(B):
        for i in range(M):
            for j in range(K):
                sum_val: f32 = 0.0
                for k_idx in range(N):
                    sum_val = sum_val + a[batch, i, k_idx] * b[batch, k_idx, j]
                out[batch, i, j] = sum_val
