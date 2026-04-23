import numpy as np
import pandas as pd

def test_matrix():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = a + b
    print(f"c = {c}")
    d = np.array([[1,2, 3], [3,4, 5], [5,6,7]])
    e = np.linalg.det(d)
    print(f"det(a) = {e}")

test_matrix()