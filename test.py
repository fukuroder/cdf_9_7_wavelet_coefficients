import unittest
import numpy as np

class TestDaubechies(unittest.TestCase):
    def test_all(self):
        scaling = np.loadtxt('cdf_9_7_scaling_coefficients.txt')
        d_scaling = np.loadtxt('cdf_9_7_dual_scaling_coefficients.txt')

        # test scaling summation
        x = np.abs(scaling.sum() - np.sqrt(2.0)) 
        self.assertLess(x, 1.0e-10, f'test scaling coefficients summation')

        # test dual scaling summation
        x = np.abs(d_scaling.sum() - np.sqrt(2.0)) 
        self.assertLess(x, 1.0e-10, f'test dual scaling coefficients summation')

        # test orthogonality
        for M in [1, 3, 5]:
            x = np.abs(np.sum([s1*s2 for s1, s2 in zip(scaling[M:], d_scaling)]))
            self.assertLess(x, 1.0e-10, f'test orthogonality M={M}')

        # test vanishing moments
        for L in range(0, 4):
            x = np.abs(np.sum([(1 if k%2==0 else -1)*s*(k**L) for k, s in enumerate(scaling)]))
            self.assertLess(x, 1.0e-10, f'test scaling vanishing moments L={L}')

        # test vanishing moments
        for L in range(0, 4):
            x = np.abs(np.sum([(1 if k%2==0 else -1)*s*(k**L) for k, s in enumerate(d_scaling)]))
            self.assertLess(x, 1.0e-10, f'test dual scaling vanishing moments L={L}')

if __name__ == '__main__':
    unittest.main()