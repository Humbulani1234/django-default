
# # test_factorial.py

# import unittest
# from factorial import factorial

# class TestFactorial(unittest.TestCase):

#     def test_factorial_of_zero(self):
#         self.assertEqual(factorial(0), 1)

#     def test_factorial_of_positive_number(self):
#         self.assertEqual(factorial(5), 120)
#         self.assertEqual(factorial(10), 3628800)

#     def test_factorial_of_negative_number(self):
#         with self.assertRaises(ValueError):
#             factorial(-1)

# if __name__ == '__main__':
#     unittest.main()


def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

# test_my_math.py

import unittest
from my_math import add, subtract

def test_add():
    # Test addition with positive numbers
    assert add(2, 3) == 5

    # Test addition with negative numbers
    assert add(-2, -3) == -5

    # Test addition of a positive and a negative number
    assert add(5, -3) == 2

def test_subtract():
    # Test subtraction with positive numbers
    assert subtract(5, 3) == 2

    # Test subtraction with negative numbers
    assert subtract(-5, -3) == -2

    # Test subtraction of a positive and a negative number
    assert subtract(5, -3) == 8

if __name__ == '__main__':
    test_add()
    test_subtract()
    print("All tests passed.")
