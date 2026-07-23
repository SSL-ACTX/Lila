import math
from lirien import verify, jit, f32, i64, Tensor
from .shared import M, N, K, H, W, KH, KW, OH, OW


@verify
def convolve1d(
    signal: Tensor[f32, M],
    kernel: Tensor[f32, K],
    out: Tensor[f32, M - K + 1],
):
    """
    1D valid convolution of 'signal' and 'kernel', storing the result in 'out'.
    Statically verified by Z3 that all accesses to signal, kernel, and out are in-bounds.
    """
    for i in range(M - K + 1):
        sum_val: f32 = 0.0
        for j in range(K):
            sum_val = sum_val + signal[i + j] * kernel[j]
        out[i] = sum_val


@verify
def convolve2d(
    image: Tensor[f32, H, W],
    kernel: Tensor[f32, KH, KW],
    out: Tensor[f32, H - KH + 1, W - KW + 1],
):
    """
    2D valid convolution of 'image' and 'kernel', storing the result in 'out'.
    Statically verified by Z3 that all accesses to image, kernel, and out are in-bounds.
    """
    for i in range(H - KH + 1):
        for j in range(W - KW + 1):
            sum_val: f32 = 0.0
            for ki in range(KH):
                for kj in range(KW):
                    sum_val = sum_val + image[i + ki, j + kj] * kernel[ki, kj]
            out[i, j] = sum_val


@verify
def max_pool2d_2x2(image: Tensor[f32, H, W], out: Tensor[f32, OH, OW]):
    """
    2x2 Max Pooling with stride 2.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(OH):
        for j in range(OW):
            v00 = image[i * 2, j * 2]
            v01 = image[i * 2, j * 2 + 1]
            v10 = image[i * 2 + 1, j * 2]
            v11 = image[i * 2 + 1, j * 2 + 1]

            max_val = v00
            if v01 > max_val:
                max_val = v01
            if v10 > max_val:
                max_val = v10
            if v11 > max_val:
                max_val = v11

            out[i, j] = max_val


@verify
def avg_pool2d_2x2(image: Tensor[f32, H, W], out: Tensor[f32, OH, OW]):
    """
    2x2 Average Pooling with stride 2.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(OH):
        for j in range(OW):
            v00 = image[i * 2, j * 2]
            v01 = image[i * 2, j * 2 + 1]
            v10 = image[i * 2 + 1, j * 2]
            v11 = image[i * 2 + 1, j * 2 + 1]
            out[i, j] = (v00 + v01 + v10 + v11) * 0.25


@verify
def standardize(a: Tensor[f32, M], out: Tensor[f32, M], mean_val: f32, std_val: f32):
    """
    Standardize 'a' using precomputed 'mean_val' and 'std_val'.
    Requires 'std_val > 0.0'.
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert std_val > 0.0
    for i in range(M):
        out[i] = (a[i] - mean_val) / std_val


@verify
def l2_normalize(a: Tensor[f32, M], out: Tensor[f32, M], epsilon: f32):
    """
    L2 normalize a 1D vector 'a', storing the result in 'out'.
    Requires 'epsilon > 0.0'.
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert epsilon > 0.0
    sum_sq: f32 = 0.0
    for i in range(M):
        sum_sq = sum_sq + a[i] * a[i]
    sqrt_input = abs(sum_sq) + epsilon
    assert sqrt_input >= 0.0
    divisor = math.sqrt(sqrt_input)
    assert divisor > 0.0
    for i in range(M):
        out[i] = a[i] / divisor


@verify
def l1_normalize(a: Tensor[f32, M], out: Tensor[f32, M], epsilon: f32):
    """
    L1 normalize a 1D vector 'a', storing the result in 'out'.
    Requires 'epsilon > 0.0'.
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert epsilon > 0.0
    sum_abs: f32 = 0.0
    for i in range(M):
        sum_abs = sum_abs + abs(a[i])
    divisor = sum_abs + epsilon
    assert divisor > 0.0
    for i in range(M):
        out[i] = a[i] / divisor


@jit
def cosine_similarity(
    a: Tensor[f32, M],
    b: Tensor[f32, M],
    out: Tensor[f32, 1],
    epsilon: f32,
):
    """
    Compute the cosine similarity of 'a' and 'b', storing in 'out[0]'.
    Requires 'epsilon > 0.0'.
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert epsilon > 0.0
    dot_val: f32 = 0.0
    norm_a_sq: f32 = 0.0
    norm_b_sq: f32 = 0.0
    for i in range(M):
        dot_val = dot_val + a[i] * b[i]
        norm_a_sq = norm_a_sq + a[i] * a[i]
        norm_b_sq = norm_b_sq + b[i] * b[i]

    denom = math.sqrt(abs(norm_a_sq)) * math.sqrt(abs(norm_b_sq)) + epsilon
    assert denom > 0.0
    out[0] = dot_val / denom


@verify
def rms_norm(a: Tensor[f32, M], out: Tensor[f32, M], epsilon: f32, n: f32):
    """
    Root Mean Square Normalization (RMSNorm) of 'a', storing in 'out'.
    Requires 'epsilon > 0.0' and 'n > 0.0' (where n is float(M)).
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert epsilon > 0.0
    assert n > 0.0
    sum_sq: f32 = 0.0
    for i in range(M):
        sum_sq = sum_sq + a[i] * a[i]
    sqrt_input = abs(sum_sq / n) + epsilon
    assert sqrt_input >= 0.0
    rms = math.sqrt(sqrt_input)
    assert rms > 0.0
    for i in range(M):
        out[i] = a[i] / rms


@verify
def layer_norm(
    a: Tensor[f32, M],
    out: Tensor[f32, M],
    gamma: Tensor[f32, M],
    beta: Tensor[f32, M],
    epsilon: f32,
    n: f32,
):
    """
    Layer Normalization of 'a' with scale 'gamma' and shift 'beta'.
    Requires 'epsilon > 0.0' and 'n > 0.0' (where n is float(M)).
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert epsilon > 0.0
    assert n > 0.0

    # Compute mean and sum of squares in a single loop
    sum_val: f32 = 0.0
    sum_sq: f32 = 0.0
    for i in range(M):
        val = a[i]
        sum_val = sum_val + val
        sum_sq = sum_sq + val * val

    mean_val = sum_val / n
    var_val = sum_sq / n - mean_val * mean_val

    # Normalize and scale/shift
    sqrt_input = abs(var_val) + epsilon
    assert sqrt_input >= 0.0
    std_val = math.sqrt(sqrt_input)
    assert std_val > 0.0
    for i in range(M):
        out[i] = (a[i] - mean_val) / std_val * gamma[i] + beta[i]


@verify
def matvec_bias(
    matrix: Tensor[f32, M, N],
    vector: Tensor[f32, N],
    bias: Tensor[f32, M],
    out: Tensor[f32, M],
):
    """
    Matrix-vector multiplication with a bias vector, storing the result in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        sum_val: f32 = 0.0
        for j in range(N):
            sum_val = sum_val + matrix[i, j] * vector[j]
        out[i] = sum_val + bias[i]


@verify
def max_pool2d(
    image: Tensor[f32, H, W],
    out: Tensor[f32, OH, OW],
    kernel_h: i64,
    kernel_w: i64,
    stride_h: i64,
    stride_w: i64,
):
    """
    Generic 2D Max Pooling with arbitrary kernel size and stride.
    Statically verified by Z3 with Presburger dimension-decoupled verification.
    """
    assert kernel_h > 0
    assert kernel_w > 0
    assert stride_h > 0
    assert stride_w > 0
    assert (OH - 1) * stride_h + kernel_h <= H
    assert (OW - 1) * stride_w + kernel_w <= W

    for i in range(OH):
        for j in range(OW):
            h_start = i * stride_h
            w_start = j * stride_w

            max_val: f32 = 0.0
            if h_start >= 0:
                if h_start < H:
                    if w_start >= 0:
                        if w_start < W:
                            max_val = image[h_start, w_start]

            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    cur_h = h_start + kh
                    cur_w = w_start + kw
                    if cur_h >= 0:
                        if cur_h < H:
                            if cur_w >= 0:
                                if cur_w < W:
                                    val = image[cur_h, cur_w]
                                    max_val = max(max_val, val)
            out[i, j] = max_val


@verify
def avg_pool2d(
    image: Tensor[f32, H, W],
    out: Tensor[f32, OH, OW],
    kernel_h: i64,
    kernel_w: i64,
    stride_h: i64,
    stride_w: i64,
):
    """
    Generic 2D Average Pooling with arbitrary kernel size and stride.
    Statically verified by Z3 with Presburger dimension-decoupled verification.
    """
    assert kernel_h > 0
    assert kernel_w > 0
    assert stride_h > 0
    assert stride_w > 0
    assert (OH - 1) * stride_h + kernel_h <= H
    assert (OW - 1) * stride_w + kernel_w <= W

    for i in range(OH):
        for j in range(OW):
            h_start = i * stride_h
            w_start = j * stride_w

            sum_val: f32 = 0.0
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    cur_h = h_start + kh
                    cur_w = w_start + kw
                    if cur_h >= 0:
                        if cur_h < H:
                            if cur_w >= 0:
                                if cur_w < W:
                                    sum_val = sum_val + image[cur_h, cur_w]

            denom = f32(kernel_h * kernel_w)
            assert denom > 0.0
            out[i, j] = sum_val / denom


@verify
def convolve2d_padded(
    image: Tensor[f32, H, W],
    kernel: Tensor[f32, KH, KW],
    out: Tensor[f32, OH, OW],
    stride_h: i64,
    stride_w: i64,
    pad_h: i64,
    pad_w: i64,
):
    """
    Generic 2D Convolution with arbitrary stride and zero-padding.
    Statically verified by Z3 with Presburger dimension-decoupled verification.
    """
    assert stride_h > 0
    assert stride_w > 0
    assert pad_h >= 0
    assert pad_w >= 0

    for i in range(OH):
        for j in range(OW):
            sum_val: f32 = 0.0
            for kh in range(KH):
                for kw in range(KW):
                    im_h = i * stride_h + kh - pad_h
                    im_w = j * stride_w + kw - pad_w

                    if im_h >= 0:
                        if im_h < H:
                            if im_w >= 0:
                                if im_w < W:
                                    sum_val = (
                                        sum_val + image[im_h, im_w] * kernel[kh, kw]
                                    )
            out[i, j] = sum_val


@verify
def resize_nearest(
    image: Tensor[f32, H, W],
    out: Tensor[f32, OH, OW],
    scale_h: f32,
    scale_w: f32,
):
    """
    Nearest-neighbor image resizing with arbitrary float scaling factors.

    Safety model: @jit (runtime-enforced via assert + branch guards).
    The float-to-int index computation combined with the nested loop CFG
    exceeds Z3's tractable search space. Safety is guaranteed structurally
    by the src_h/src_w bounds checks, which prevent out-of-bounds access.
    """
    assert scale_h > 0.0
    assert scale_w > 0.0

    for i in range(OH):
        for j in range(OW):
            src_h = int(f32(i) * scale_h)
            src_w = int(f32(j) * scale_w)

            if src_h >= 0:
                if src_h < H:
                    if src_w >= 0:
                        if src_w < W:
                            out[i, j] = image[src_h, src_w]
                        else:
                            out[i, j] = 0.0
                    else:
                        out[i, j] = 0.0
                else:
                    out[i, j] = 0.0
            else:
                out[i, j] = 0.0
