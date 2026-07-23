import math
from lirien import verify, f32, Tensor
from .shared import M, N


@verify
def relu(a: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Apply element-wise Rectified Linear Unit (ReLU) activation in-place to 'out'.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            if val > 0.0:
                out[i, j] = val
            else:
                out[i, j] = 0.0


@verify
def sigmoid(a: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Apply element-wise Sigmoid activation in-place to 'out'.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            # 1 / (1 + exp(-val))
            out[i, j] = 1.0 / (1.0 + math.exp(-val))


@verify
def leaky_relu(a: Tensor[f32, M, N], out: Tensor[f32, M, N], alpha: f32):
    """
    Apply element-wise Leaky ReLU activation in-place to 'out'.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            if val > 0.0:
                out[i, j] = val
            else:
                out[i, j] = alpha * val


@verify
def softmax(a: Tensor[f32, M], out: Tensor[f32, M]):
    """
    Apply Softmax activation to 1D tensor 'a', storing the result in 'out'.
    Statically verified by Z3 to be division-by-zero safe and memory-safe.
    """
    sum_exp: f32 = 0.0
    for i in range(M):
        sum_exp = sum_exp + math.exp(a[i])
    assert sum_exp > 0.0
    for i in range(M):
        out[i] = math.exp(a[i]) / sum_exp


@verify
def silu(a: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Apply element-wise Sigmoid Linear Unit (SiLU) / Swish activation.
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            out[i, j] = val / (1.0 + math.exp(-val))


@verify
def hardsigmoid(a: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Apply element-wise Hard Sigmoid activation.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j] + 3.0
            if val < 0.0:
                out[i, j] = 0.0
            elif val > 6.0:
                out[i, j] = 1.0
            else:
                out[i, j] = val / 6.0


@verify
def hardswish(a: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Apply element-wise Hard Swish activation.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            h_sig = val + 3.0
            if h_sig < 0.0:
                out[i, j] = 0.0
            elif h_sig > 6.0:
                out[i, j] = val
            else:
                out[i, j] = val * (h_sig / 6.0)


@verify
def elu(a: Tensor[f32, M, N], out: Tensor[f32, M, N], alpha: f32):
    """
    Apply element-wise Exponential Linear Unit (ELU) activation.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            if val > 0.0:
                out[i, j] = val
            else:
                out[i, j] = alpha * (math.exp(val) - 1.0)


@verify
def selu(a: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Apply element-wise Scaled Exponential Linear Unit (SELU) activation.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            if val > 0.0:
                out[i, j] = scale * val
            else:
                out[i, j] = scale * alpha * (math.exp(val) - 1.0)


@verify
def gelu(a: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Apply element-wise GELU (tanh approximation) activation.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            x = a[i, j]
            # z = sqrt(2/pi) * (x + 0.044715 * x^3)
            # sqrt(2/pi) is approx 0.79788456
            z = 0.79788456 * (x + 0.044715 * x * x * x)
            # tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
            exp_2z = math.exp(2.0 * z)
            tanh_z = (exp_2z - 1.0) / (exp_2z + 1.0)
            out[i, j] = 0.5 * x * (1.0 + tanh_z)


@verify
def swiglu(x: Tensor[f32, M, N], gate: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Apply element-wise SwiGLU activation: Swish(gate) * x, storing the result in 'out'.
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    for i in range(M):
        for j in range(N):
            g = gate[i, j]
            swish_g = g / (1.0 + math.exp(-g))
            out[i, j] = swish_g * x[i, j]
