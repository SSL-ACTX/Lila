import unittest
import math
from lirien import Tensor, f32, num


class TestStdlibTraining(unittest.TestCase):
    def test_sigmoid_cross_entropy(self):
        logits = Tensor.alloc((2, 2), f32)
        targets = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        logits[0, 0] = 0.0
        targets[0, 0] = 0.5
        logits[0, 1] = 1.0
        targets[0, 1] = 1.0
        logits[1, 0] = -2.0
        targets[1, 0] = 0.0
        logits[1, 1] = 10.0
        targets[1, 1] = 0.0

        num.sigmoid_cross_entropy(logits, targets, out)

        self.assertAlmostEqual(out[0, 0], 0.693147, places=5)
        self.assertAlmostEqual(out[0, 1], 0.3132617, places=5)
        self.assertAlmostEqual(out[1, 0], 0.126928, places=5)
        self.assertAlmostEqual(out[1, 1], 10.000045, places=5)

    def test_l2_loss(self):
        a = Tensor.alloc((2, 2), f32)
        b = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((1,), f32)

        a[0, 0] = 1.0
        a[0, 1] = 2.0
        a[1, 0] = 3.0
        a[1, 1] = 4.0

        b[0, 0] = 2.0
        b[0, 1] = 1.0
        b[1, 0] = 4.0
        b[1, 1] = 2.0

        num.l2_loss(a, b, out, 8.0)

        self.assertAlmostEqual(out[0], 0.875, places=5)

    def test_sgd_momentum(self):
        param = Tensor.alloc((2, 2), f32)
        grad = Tensor.alloc((2, 2), f32)
        velocity = Tensor.alloc((2, 2), f32)

        param[0, 0] = 1.0
        param[0, 1] = 2.0
        param[1, 0] = 3.0
        param[1, 1] = 4.0

        grad[0, 0] = 0.1
        grad[0, 1] = 0.2
        grad[1, 0] = 0.3
        grad[1, 1] = 0.4

        velocity[0, 0] = 0.01
        velocity[0, 1] = 0.02
        velocity[1, 0] = 0.03
        velocity[1, 1] = 0.04

        num.sgd_momentum_step(param, grad, velocity, 0.1, 0.9)

        self.assertAlmostEqual(velocity[0, 0], 0.9 * 0.01 + 0.1 * 0.1, places=5)
        self.assertAlmostEqual(param[0, 0], 1.0 - (0.9 * 0.01 + 0.1 * 0.1), places=5)

    def test_adamw(self):
        param = Tensor.alloc((2, 2), f32)
        grad = Tensor.alloc((2, 2), f32)
        m = Tensor.alloc((2, 2), f32)
        v = Tensor.alloc((2, 2), f32)

        param[0, 0] = 1.0
        param[0, 1] = 2.0
        grad[0, 0] = 0.1
        grad[0, 1] = 0.2
        m[0, 0] = 0.01
        m[0, 1] = 0.02
        v[0, 0] = 0.001
        v[0, 1] = 0.002

        num.adamw_step(param, grad, m, v, 0.001, 0.9, 0.999, 1e-8, 0.01, 0.9, 0.99)

        self.assertAlmostEqual(m[0, 0], 0.019, places=5)
        self.assertAlmostEqual(v[0, 0], 0.001009, places=5)
        self.assertAlmostEqual(param[0, 0], 0.9993288, places=5)

    def test_softmax_cross_entropy(self):
        logits = Tensor.alloc((2, 3), f32)
        targets = Tensor.alloc((2, 3), f32)
        out = Tensor.alloc((2,), f32)

        logits[0, 0] = 1.0
        logits[0, 1] = 2.0
        logits[0, 2] = 3.0
        logits[1, 0] = 0.5
        logits[1, 1] = 1.5
        logits[1, 2] = 0.1

        targets[0, 0] = 0.0
        targets[0, 1] = 0.0
        targets[0, 2] = 1.0
        targets[1, 0] = 0.2
        targets[1, 1] = 0.8
        targets[1, 2] = 0.0

        num.softmax_cross_entropy_with_logits(logits, targets, out)

        def ref_ce(log_vals, target_vals):
            max_v = max(log_vals)
            lse = max_v + math.log(sum(math.exp(x - max_v) for x in log_vals))
            return sum(t * (lse - x) for t, x in zip(target_vals, log_vals))

        self.assertAlmostEqual(
            out[0], ref_ce([1.0, 2.0, 3.0], [0.0, 0.0, 1.0]), places=5
        )
        self.assertAlmostEqual(
            out[1], ref_ce([0.5, 1.5, 0.1], [0.2, 0.8, 0.0]), places=5
        )


if __name__ == "__main__":
    unittest.main()
