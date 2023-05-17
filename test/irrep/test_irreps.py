import unittest
import su2nn.irrep as irrep

class Test_Irrep(unittest.TestCase):
    def test_init(self):
        self.assertEqual(str(irrep.Irrep(1, 1, 1)), '1ee')

if __name__ == '__main__':
    unittest.main()