import unittest
from su2nn.irrep import Irrep

class Test_Irrep(unittest.TestCase):
    def test_init(self):
        print('Test initializing Irrep')
        self.assertEqual(str(Irrep(1, 1, 1)), '1ee')
        self.assertEqual(str(Irrep(0.5, 1, -1)), '1/2eo')
        self.assertEqual(str(Irrep(1.5, 'e', -1)), '3/2eo')
        self.assertEqual(str(Irrep('1/2oe')), '1/2oe')
        self.assertRaises(ValueError, Irrep, *(1.6, 1, 1))
        self.assertRaises(ValueError, Irrep, *(1, 1, 2))
        self.assertRaises(ValueError, Irrep, *(1, 2, 1))


if __name__ == '__main__':
    unittest.main()