import unittest
import math
from lirien import Tensor, f32, num


class TestStdlibActivations(unittest.TestCase):
    def test_relu(self):
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 1.5
        a[0, 1] = -2.0
        a[1, 0] = 0.0
        a[1, 1] = -0.5

        num.relu(a, out)

        self.assertEqual(out[0, 0], 1.5)
        self.assertEqual(out[0, 1], 0.0)
        self.assertEqual(out[1, 0], 0.0)
        self.assertEqual(out[1, 1], 0.0)

    def test_leaky_relu(self):
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 1.5
        a[0, 1] = -2.0
        a[1, 0] = 0.0
        a[1, 1] = -0.5

        num.leaky_relu(a, out, 0.1)

        self.assertEqual(out[0, 0], 1.5)
        self.assertAlmostEqual(out[0, 1], -0.2)
        self.assertEqual(out[1, 0], 0.0)
        self.assertAlmostEqual(out[1, 1], -0.05)

    def test_sigmoid(self):
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 0.0
        a[0, 1] = 1.0
        a[1, 0] = -1.0
        a[1, 1] = 10.0

        num.sigmoid(a, out)

        self.assertAlmostEqual(out[0, 0], 0.5)
        self.assertAlmostEqual(out[0, 1], 1.0 / (1.0 + math.exp(-1.0)))
        self.assertAlmostEqual(out[1, 0], 1.0 / (1.0 + math.exp(1.0)))
        self.assertAlmostEqual(out[1, 1], 1.0 / (1.0 + math.exp(-10.0)))

    def test_silu(self):
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 0.0
        a[0, 1] = 1.0
        a[1, 0] = -1.0
        a[1, 1] = 2.0

        num.silu(a, out)

        self.assertAlmostEqual(out[0, 0], 0.0)
        self.assertAlmostEqual(out[0, 1], 1.0 / (1.0 + math.exp(-1.0)), places=5)
        self.assertAlmostEqual(out[1, 0], -1.0 / (1.0 + math.exp(1.0)), places=5)
        self.assertAlmostEqual(out[1, 1], 2.0 / (1.0 + math.exp(-2.0)), places=5)

    def test_hardsigmoid(self):
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = -4.0
        a[0, 1] = 0.0
        a[1, 0] = 3.0
        a[1, 1] = -1.5

        num.hardsigmoid(a, out)

        self.assertAlmostEqual(out[0, 0], 0.0)
        self.assertAlmostEqual(out[0, 1], 0.5)
        self.assertAlmostEqual(out[1, 0], 1.0)
        self.assertAlmostEqual(out[1, 1], 0.25)

    def test_hardswish(self):
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = -4.0
        a[0, 1] = 0.0
        a[1, 0] = 3.0
        a[1, 1] = -1.5

        num.hardswish(a, out)

        self.assertAlmostEqual(out[0, 0], 0.0)
        self.assertAlmostEqual(out[0, 1], 0.0)
        self.assertAlmostEqual(out[1, 0], 3.0)
        self.assertAlmostEqual(out[1, 1], -0.375)

    def test_elu(self):
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 1.0
        a[0, 1] = -1.0
        a[1, 0] = 0.0
        a[1, 1] = -2.0

        num.elu(a, out, 1.0)

        self.assertAlmostEqual(out[0, 0], 1.0)
        self.assertAlmostEqual(out[0, 1], math.exp(-1.0) - 1.0, places=5)
        self.assertAlmostEqual(out[1, 0], 0.0)
        self.assertAlmostEqual(out[1, 1], math.exp(-2.0) - 1.0, places=5)

    def test_selu(self):
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 1.0
        a[0, 1] = -1.0
        a[1, 0] = 0.0
        a[1, 1] = -2.0

        num.selu(a, out)

        scale = 1.0507009873554804934193349852946
        alpha = 1.6732632423543772848170429916717

        self.assertAlmostEqual(out[0, 0], scale * 1.0, places=5)
        self.assertAlmostEqual(
            out[0, 1], scale * alpha * (math.exp(-1.0) - 1.0), places=5
        )
        self.assertAlmostEqual(out[1, 0], 0.0, places=5)
        self.assertAlmostEqual(
            out[1, 1], scale * alpha * (math.exp(-2.0) - 1.0), places=5
        )

    def test_gelu(self):
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 0.0
        a[0, 1] = 1.0
        a[1, 0] = -1.0
        a[1, 1] = 2.5

        num.gelu(a, out)

        def gelu_ref(x):
            z = 0.79788456 * (x + 0.044715 * x * x * x)
            tanh_z = math.tanh(z)
            return 0.5 * x * (1.0 + tanh_z)

        self.assertAlmostEqual(out[0, 0], gelu_ref(0.0), places=5)
        self.assertAlmostEqual(out[0, 1], gelu_ref(1.0), places=5)
        self.assertAlmostEqual(out[1, 0], gelu_ref(-1.0), places=5)
        self.assertAlmostEqual(out[1, 1], gelu_ref(2.5), places=5)

    def test_swiglu(self):
        x = Tensor.alloc((2, 2), f32)
        gate = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        x[0, 0] = 1.5
        x[0, 1] = -2.0
        x[1, 0] = 0.5
        x[1, 1] = -1.0

        gate[0, 0] = 0.0
        gate[0, 1] = 1.0
        gate[1, 0] = -1.0
        gate[1, 1] = 2.0

        num.swiglu(x, gate, out)

        def swiglu_ref(x_val, g_val):
            silu_g = g_val / (1.0 + math.exp(-g_val))
            return silu_g * x_val

        self.assertAlmostEqual(out[0, 0], swiglu_ref(1.5, 0.0), places=5)
        self.assertAlmostEqual(out[0, 1], swiglu_ref(-2.0, 1.0), places=5)
        self.assertAlmostEqual(out[1, 0], swiglu_ref(0.5, -1.0), places=5)
        self.assertAlmostEqual(out[1, 1], swiglu_ref(-1.0, 2.0), places=5)


if __name__ == "__main__":
    unittest.main()
