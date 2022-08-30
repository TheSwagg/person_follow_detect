import numpy as np
from utils_ import convert_position_to_state

class kalmanFilter:
    def __init__(self):     # 初始化卡尔曼滤波
        # ------initialized data------
        # 处理噪声协方差矩阵
        self.q = np.zeros((6, 6))
        # 测量噪声协方差矩阵
        self.r = np.zeros((6, 6))
        # 状态转移矩阵
        self.a = np.zeros((6, 6))
        # 测量矩阵
        self.h = np.zeros((6, 6))
        # 单位矩阵
        self.i = np.zeros((6, 6))

        # ------current data------
        # measurement
        self.measurement = np.zeros(6)
        # x的先验估计值
        self.x_prior_estimate = np.zeros(6)
        # p的先验估计值
        self.p_prior_estimate = np.zeros(6)
        # 卡尔曼滤波增益
        self.k = np.zeros((6, 6))
        # x的后验估计值
        self.x_posterior_estimate = np.zeros(6)
        # p的后验估计值
        self.p_posterior_estimate = np.zeros((6, 6))

        # ------last data------
        # last position
        self.last_position = np.zeros(4)
        # x的最后后验估计值
        self.last_x_posterior_estimate = np.zeros(6)
        # p的最后后验估计值
        self.last_p_posterior_estimate = np.zeros((6, 6))

    def initialize(self):
        # ------initialized data------
        self.q = np.eye(6) * 0.1
        self.r = np.eye(6)
        self.a = np.array([[1, 0, 0, 0, 1, 0],
                           [0, 1, 0, 0, 0, 1],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        self.h = np.eye(6)
        self.i = np.eye(6)

        # ------last data------
        self.last_position = np.array([607, 200, 635, 282])
        self.last_x_posterior_estimate = convert_position_to_state(self.last_position)
        self.last_p_posterior_estimate = np.eye(6)