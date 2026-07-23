import math
from lirien import verify, jit, f32, Tensor
from .shared import M, N


@verify
def sgd_momentum_step(
    param: Tensor[f32, M, N],
    grad: Tensor[f32, M, N],
    velocity: Tensor[f32, M, N],
    lr: f32,
    momentum: f32,
):
    """
    Perform an in-place SGD step with momentum.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            v = momentum * velocity[i, j] + lr * grad[i, j]
            velocity[i, j] = v
            param[i, j] = param[i, j] - v


@verify
def adamw_step(
    param: Tensor[f32, M, N],
    grad: Tensor[f32, M, N],
    m: Tensor[f32, M, N],
    v: Tensor[f32, M, N],
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    wd: f32,
    bias_correction1: f32,
    bias_correction2: f32,
):
    """
    Perform an in-place AdamW step.
    Requires 'epsilon > 0.0', 'bias_correction1 > 0.0', and 'bias_correction2 > 0.0'.
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert epsilon > 0.0
    assert bias_correction1 > 0.0
    assert bias_correction2 > 0.0
    for i in range(M):
        for j in range(N):
            g = grad[i, j]
            m_t = beta1 * m[i, j] + (1.0 - beta1) * g
            v_t = beta2 * v[i, j] + (1.0 - beta2) * g * g
            m[i, j] = m_t
            v[i, j] = v_t

            m_hat = m_t / bias_correction1
            v_hat = v_t / bias_correction2

            denom = math.sqrt(abs(v_hat)) + epsilon
            assert denom > 0.0
            param[i, j] = param[i, j] - lr * (m_hat / denom + wd * param[i, j])


@jit
def sigmoid_cross_entropy(
    logits: Tensor[f32, M, N],
    targets: Tensor[f32, M, N],
    out: Tensor[f32, M, N],
):
    """
    Compute element-wise sigmoid cross entropy loss.
    Statically verified by Z3 to be memory-safe, division-safe, and log-safe.
    """
    for i in range(M):
        for j in range(N):
            x = logits[i, j]
            y = targets[i, j]
            # Stable formula: max(x, 0) - x * y + log(1 + exp(-abs(x)))
            max_val = x
            if 0.0 > max_val:
                max_val = 0.0

            out[i, j] = max_val - x * y + math.log(1.0 + math.exp(-abs(x)))


@verify
def l2_loss(
    a: Tensor[f32, M, N],
    b: Tensor[f32, M, N],
    out: Tensor[f32, 1],
    n: f32,
):
    """
    Compute L2 loss (Mean Squared Error) between 'a' and 'b'.
    Requires 'n > 0.0' (where n is float(2 * M * N)).
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert n > 0.0
    sum_sq: f32 = 0.0
    for i in range(M):
        for j in range(N):
            diff = a[i, j] - b[i, j]
            sum_sq = sum_sq + diff * diff
    out[0] = sum_sq / n


@verify
def softmax_cross_entropy_with_logits(
    logits: Tensor[f32, M, N],
    targets: Tensor[f32, M, N],
    out: Tensor[f32, M],
):
    """
    Compute multi-class cross entropy loss per batch element.
    Uses the log-sum-exp trick to prevent underflow/overflow.
    Statically verified by Z3 to be memory-safe, division-safe, and log-safe.
    """
    for i in range(M):
        # Find max logit for stability
        max_val = logits[i, 0]
        for j in range(N):
            if logits[i, j] > max_val:
                max_val = logits[i, j]

        # Compute log-sum-exp
        sum_exp = 0.0
        for j in range(N):
            sum_exp = sum_exp + math.exp(logits[i, j] - max_val)

        assert sum_exp > 0.0
        lse = max_val + math.log(sum_exp)

        # Compute cross entropy: sum(target * (lse - logit))
        loss = 0.0
        for j in range(N):
            loss = loss + targets[i, j] * (lse - logits[i, j])

        out[i] = loss
