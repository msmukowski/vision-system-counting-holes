import unittest
import numpy as np
from vsch.helpers.processing import Picture as pic

class TestClass(unittest.TestCase):
    def test_line_width(self):
        p_1 = pic("")
        p_2 = pic("")

        self.assertEqual(p_1.line_width(0,0,1,1), np.sqrt(2))
        self.assertEqual(p_2.line_width(2,3,0,0),np.sqrt(13))

if __name__ == '__main__':
    unittest.main()