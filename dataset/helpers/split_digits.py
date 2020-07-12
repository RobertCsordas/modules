import numpy as np


def split_digits(n_digits: int, nums: np.ndarray) -> np.ndarray:
    digits = []
    for d in range(n_digits):
        digits.append(nums % 10)
        nums = nums // 10

    return np.stack(digits, -1).astype(np.uint8)
