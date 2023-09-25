
# test_my_math.py

import unittest
from operators import add, subtract

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
