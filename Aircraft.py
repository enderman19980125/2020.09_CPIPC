import numpy as np


class Aircraft:
    def __init__(self):
        self.barycenter = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.oil_tank_position = np.array([
            [8.91304348, 1.20652174, 0.61669004],
            [6.91304348, -1.39347826, 0.21669004],
            [-1.68695652, 1.20652174, -0.28330996],
            [3.11304348, 0.60652174, -0.18330996],
            [-5.28695652, -0.29347826, 0.41669004],
            [-2.08695652, -1.49347826, 0.21669004]
        ], dtype=np.float32)
        self.oil_tank_size = np.array([
            [1.5, 0.9, 0.3],
            [2.2, 0.8, 1.1],
            [2.4, 1.1, 0.9],
            [1.7, 1.3, 1.2],
            [2.4, 1.2, 1],
            [2.4, 1, 0.5],
        ], dtype=np.float32)
        self.oil_tank_init_oil = np.array([
            0.3, 1.5, 2.1, 1.9, 2.6, 0.8
        ], dtype=np.float32)
