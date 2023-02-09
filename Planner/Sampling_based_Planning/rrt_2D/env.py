"""
Environment for rrt_2D
@author: huiming zhou
"""


class Env:
    def __init__(self):
        #self.x_range = (0, 50)
        #self.y_range = (0, 30)
        self.x_range = (0, 99)
        self.y_range = (0, 99)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    @staticmethod
    def obs_boundary():
        # obs_boundary = [
        #     [0, 0, 1, 30],
        #     [0, 30, 50, 1],
        #     [1, 0, 50, 1],
        #     [50, 1, 1, 30]
        # ]
        obs_boundary = [
            [0, 0, 1, 99],
            [0, 99, 99, 1],
            [1, 0, 99, 1],
            [99, 1, 1, 99]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        # obs_rectangle = [
        #     [14, 12, 8, 2],
        #     [18, 22, 8, 3],
        #     [26, 7, 2, 12],
        #     [32, 14, 10, 2]
        # ]
        obs_rectangle = []
        return obs_rectangle

    @staticmethod
    def obs_circle():
        # obs_cir = [
        #     [7, 12, 3],
        #     [46, 20, 2],
        #     [15, 5, 2],
        #     [37, 7, 3],
        #     [37, 23, 3]
        # ]
        obs_cir = []

        return obs_cir
