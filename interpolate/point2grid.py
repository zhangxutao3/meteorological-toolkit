import numpy as np
import torch
import cupy as cp

"""
这都是我的心血啊
"""


class IDW:

    def __init__(self, lon_obs, lat_obs, value_obs, lon_to, lat_to, device="cpu", gpu_id=0):

        # 示例用法
        # idw = IDW(lon_obs, lat_obs, value_obs, lon_to, lat_to, device="cpu")  # 或者 device="cuda:0"

        self.device = device

        # 将输入数据转换为 numpy 数组
        if device == "cpu":
            self.lon_obs = np.array(lon_obs, dtype=np.float32)
            self.lat_obs = np.array(lat_obs, dtype=np.float32)
            self.value_obs = np.array(value_obs, dtype=np.float32)
            self.lon_to = np.array(lon_to, dtype=np.float32)
            self.lat_to = np.array(lat_to, dtype=np.float32)
        elif device == "cuda":
            self.lon_obs = cp.array(lon_obs, dtype=cp.float32)
            self.lat_obs = cp.array(lat_obs, dtype=cp.float32)
            self.value_obs = cp.array(value_obs, dtype=cp.float32)
            self.lon_to = cp.array(lon_to, dtype=cp.float32)
            self.lat_to = cp.array(lat_to, dtype=cp.float32)

        # 检查设备是否为 GPU，并且 CUDA 是否可用
        if self.device=="cpu":
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA and CuPy is not available on this system. Please check your CUDA installation.")
        else:
            device = cp.cuda.Device(gpu_id)
            device.use()  # 切换到指定的 GPU


        # 检查是否存在 NaN 值
        if np.isnan(self.value_obs).any():
            raise ValueError("Input values contain NaN. Please provide valid data.")

        # 网格点
        lon_grid, lat_grid = np.meshgrid(self.lon_to, self.lat_to) if device=="cpu" else cp.meshgrid(self.lon_to, self.lat_to)
        self.grid_points = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T if device=="cpu" else cp.vstack([lon_grid.ravel(), lat_grid.ravel()]).T

        # 根据设备选择插值方法
        if self.device == "cpu":
            self.interpolate = self._interpolate_cpu
        elif self.device.startswith('cuda'):
            self.interpolate = self._interpolate_gpu
        else:
            raise ValueError(f"Unsupported device: {self.device}")

    def _interpolate_cpu(self):
        # CPU版本的插值实现
        dist_matrix = self.compute_distance_matrix(np.vstack([self.lon_obs, self.lat_obs]).T, self.grid_points)
        weights = 1 / (dist_matrix ** 2)
        weights[dist_matrix == 0] = np.inf

        results = np.dot(weights.T, self.value_obs) / np.sum(weights, axis=0)
        return results.reshape(self.lat_to.size, self.lon_to.size)

    def _interpolate_gpu(self):
        # gpu 版本插值实现
        dist_matrix = self.compute_distance_matrix(cp.vstack([self.lon_obs, self.lat_obs]).T, self.grid_points)
        weights = 1 / (dist_matrix ** 2)
        weights[weights == 0] = cp.inf  # 防止除以零

        results = cp.dot(weights.T, self.value_obs) / cp.sum(weights, axis=0)

        return results.reshape(self.lat_to.size, self.lon_to.size).get()

    def compute_distance_matrix(self, points, grid_points, ):
        """
        使用CuPy计算点与网格之间的欧几里得距离矩阵
        """
        # 广播计算所有点与所有网格点之间的距离
        diff = points[:, None, :] - grid_points[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)  if self.device == "cpu" else cp.linalg.norm(diff, axis=-1)
        return dist_matrix