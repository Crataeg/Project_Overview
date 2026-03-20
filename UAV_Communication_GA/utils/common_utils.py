"""
通用工具函数
提供数学计算、单位转换、几何计算等通用功能
支持无人机通信优化系统的3D计算需求
"""

import numpy as np
import math
from typing import Union, List, Tuple, Any, Optional


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """
    归一化数值到[0,1]范围

    Args:
        value: 待归一化的数值
        min_val: 最小值
        max_val: 最大值

    Returns:
        归一化后的数值
    """
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def denormalize_value(norm_value: float, min_val: float, max_val: float) -> float:
    """
    反归一化数值

    Args:
        norm_value: 归一化后的数值 [0,1]
        min_val: 最小值
        max_val: 最大值

    Returns:
        反归一化后的数值
    """
    return min_val + norm_value * (max_val - min_val)


def bound_value(value: float, lower: float, upper: float) -> float:
    """
    限制数值在指定范围内

    Args:
        value: 待限制的数值
        lower: 下限
        upper: 上限

    Returns:
        限制后的数值
    """
    return max(lower, min(upper, value))


def linear_interpolation(x: float, x1: float, x2: float, y1: float, y2: float) -> float:
    """
    线性插值

    Args:
        x: 待插值点
        x1, x2: 已知x坐标
        y1, y2: 已知y坐标

    Returns:
        插值结果
    """
    if x2 == x1:
        return (y1 + y2) / 2
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def calculate_distance(pos1: Union[List[float], Tuple[float, ...], np.ndarray],
                       pos2: Union[List[float], Tuple[float, ...], np.ndarray]) -> float:
    """
    计算两点之间的欧氏距离 (支持2D和3D)

    Args:
        pos1: 第一个点的坐标 [x, y] 或 [x, y, z]
        pos2: 第二个点的坐标 [x, y] 或 [x, y, z]

    Returns:
        两点之间的欧氏距离
    """
    pos1_arr = np.array(pos1)
    pos2_arr = np.array(pos2)

    # 确保维度一致
    if len(pos1_arr) != len(pos2_arr):
        min_dim = min(len(pos1_arr), len(pos2_arr))
        pos1_arr = pos1_arr[:min_dim]
        pos2_arr = pos2_arr[:min_dim]

    return np.sqrt(np.sum((pos1_arr - pos2_arr) ** 2))


def calculate_euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    计算欧氏距离 (兼容不同维度的点)

    Args:
        point1: 第一个点
        point2: 第二个点

    Returns:
        欧氏距离
    """
    # 确保维度一致
    if len(point1) != len(point2):
        min_dim = min(len(point1), len(point2))
        point1 = point1[:min_dim]
        point2 = point2[:min_dim]

    return np.linalg.norm(point1 - point2)


def calculate_manhattan_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    计算曼哈顿距离

    Args:
        point1: 第一个点
        point2: 第二个点

    Returns:
        曼哈顿距离
    """
    # 确保维度一致
    if len(point1) != len(point2):
        min_dim = min(len(point1), len(point2))
        point1 = point1[:min_dim]
        point2 = point2[:min_dim]

    return np.sum(np.abs(point1 - point2))


def calculate_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两个向量之间的夹角 (弧度)

    Args:
        v1: 第一个向量
        v2: 第二个向量

    Returns:
        夹角 (弧度)
    """
    # 确保维度一致
    if len(v1) != len(v2):
        min_dim = min(len(v1), len(v2))
        v1 = v1[:min_dim]
        v2 = v2[:min_dim]

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免浮点误差
    return np.arccos(cos_angle)


def db_to_linear(db_value: float) -> float:
    """
    dB值转换为线性值

    Args:
        db_value: dB值

    Returns:
        线性值
    """
    return 10 ** (db_value / 10)


def linear_to_db(linear_value: float) -> float:
    """
    线性值转换为dB值

    Args:
        linear_value: 线性值

    Returns:
        dB值
    """
    if linear_value <= 0:
        return -np.inf
    return 10 * np.log10(linear_value)


def gaussian_function(x: float, mean: float, std: float) -> float:
    """
    高斯函数

    Args:
        x: 输入值
        mean: 均值
        std: 标准差

    Returns:
        高斯函数值
    """
    return np.exp(-0.5 * ((x - mean) / std) ** 2)


def sigmoid_function(x: float, center: float, slope: float) -> float:
    """
    Sigmoid函数

    Args:
        x: 输入值
        center: 中心点
        slope: 斜率

    Returns:
        Sigmoid函数值
    """
    return 1 / (1 + np.exp(-slope * (x - center)))


def logistic_function(x: float, L: float = 1.0, k: float = 1.0, x0: float = 0.0) -> float:
    """
    逻辑斯蒂函数

    Args:
        x: 输入值
        L: 最大值
        k: 生长率
        x0: 中心点

    Returns:
        逻辑斯蒂函数值
    """
    return L / (1 + np.exp(-k * (x - x0)))


def check_position_in_range(position: Union[List[float], Tuple[float, ...], np.ndarray],
                            range_bounds: Tuple[float, float, float, float, float, float]) -> bool:
    """
    检查3D位置是否在指定范围内

    Args:
        position: 位置坐标 [x, y, z]
        range_bounds: 范围边界 [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns:
        是否在范围内
    """
    if len(position) == 2:  # 2D位置
        x, y = position[0], position[1]
        x_min, y_min, z_min, x_max, y_max, z_max = range_bounds
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)
    elif len(position) == 3:  # 3D位置
        x, y, z = position[0], position[1], position[2]
        x_min, y_min, z_min, x_max, y_max, z_max = range_bounds
        return (x_min <= x <= x_max) and (y_min <= y <= y_max) and (z_min <= z <= z_max)
    else:
        return False


def generate_random_positions(num_positions: int,
                              x_range: Tuple[float, float] = (0, 1000),
                              y_range: Tuple[float, float] = (0, 1000),
                              z_range: Tuple[float, float] = (0, 100)) -> np.ndarray:
    """
    生成3D随机位置

    Args:
        num_positions: 位置数量
        x_range: x坐标范围
        y_range: y坐标范围
        z_range: z坐标范围

    Returns:
        3D随机位置数组 [num_positions, 3]
    """
    positions = np.zeros((num_positions, 3))
    positions[:, 0] = np.random.uniform(x_range[0], x_range[1], num_positions)
    positions[:, 1] = np.random.uniform(y_range[0], y_range[1], num_positions)
    positions[:, 2] = np.random.uniform(z_range[0], z_range[1], num_positions)
    return positions


def calculate_centroid(positions: np.ndarray) -> np.ndarray:
    """
    计算3D位置集合的质心

    Args:
        positions: 3D位置数组 [n, 3]

    Returns:
        质心坐标 [x, y, z]
    """
    return np.mean(positions, axis=0)


def calculate_bounding_box(positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算3D位置集合的包围盒

    Args:
        positions: 3D位置数组 [n, 3]

    Returns:
        (最小点, 最大点)
    """
    min_point = np.min(positions, axis=0)
    max_point = np.max(positions, axis=0)
    return min_point, max_point


def calculate_area(coordinates: List[Tuple[float, float]]) -> float:
    """
    计算多边形面积 (鞋带公式)

    Args:
        coordinates: 多边形顶点坐标列表

    Returns:
        多边形面积
    """
    if len(coordinates) < 3:
        return 0.0

    area = 0.0
    n = len(coordinates)

    for i in range(n):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)

    return abs(area) / 2.0


def radians_to_degrees(radians: float) -> float:
    """
    弧度转角度

    Args:
        radians: 弧度值

    Returns:
        角度值
    """
    return radians * 180 / math.pi


def degrees_to_radians(degrees: float) -> float:
    """
    角度转弧度

    Args:
        degrees: 角度值

    Returns:
        弧度值
    """
    return degrees * math.pi / 180


def calculate_circle_area(radius: float) -> float:
    """
    计算圆面积

    Args:
        radius: 半径

    Returns:
        圆面积
    """
    return math.pi * radius ** 2


def calculate_sphere_volume(radius: float) -> float:
    """
    计算球体积

    Args:
        radius: 半径

    Returns:
        球体积
    """
    return (4 / 3) * math.pi * radius ** 3


def calculate_triangle_area(a: float, b: float, c: float) -> float:
    """
    计算三角形面积 (海伦公式)

    Args:
        a, b, c: 三角形三边长度

    Returns:
        三角形面积
    """
    s = (a + b + c) / 2
    return math.sqrt(s * (s - a) * (s - b) * (s - c))


def is_point_in_triangle(point: Tuple[float, float],
                         triangle: List[Tuple[float, float]]) -> bool:
    """
    判断点是否在三角形内 (重心坐标法)

    Args:
        point: 待判断的点
        triangle: 三角形三个顶点

    Returns:
        是否在三角形内
    """
    if len(triangle) != 3:
        return False

    A, B, C = triangle
    P = point

    # 计算重心坐标
    denominator = ((B[1] - C[1]) * (A[0] - C[0]) +
                   (C[0] - B[0]) * (A[1] - C[1]))

    if denominator == 0:
        return False

    alpha = ((B[1] - C[1]) * (P[0] - C[0]) +
             (C[0] - B[0]) * (P[1] - C[1])) / denominator

    beta = ((C[1] - A[1]) * (P[0] - C[0]) +
            (A[0] - C[0]) * (P[1] - C[1])) / denominator

    gamma = 1 - alpha - beta

    # 检查重心坐标是否都在[0,1]范围内
    return 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1


def calculate_line_equation(point1: Tuple[float, float],
                            point2: Tuple[float, float]) -> Tuple[float, float]:
    """
    计算直线方程 y = mx + b

    Args:
        point1: 第一个点
        point2: 第二个点

    Returns:
        (斜率m, 截距b)
    """
    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2:
        return float('inf'), x1  # 垂直线

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    return m, b


def calculate_perpendicular_distance(point: Tuple[float, float],
                                     line_point1: Tuple[float, float],
                                     line_point2: Tuple[float, float]) -> float:
    """
    计算点到直线的垂直距离

    Args:
        point: 点坐标
        line_point1: 直线上第一个点
        line_point2: 直线上第二个点

    Returns:
        垂直距离
    """
    x, y = point
    x1, y1 = line_point1
    x2, y2 = line_point2

    if x1 == x2 and y1 == y2:
        return calculate_distance(point, line_point1)

    numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    return numerator / denominator


def calculate_3d_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    计算3D空间中的距离

    Args:
        pos1: 第一个3D点 [x, y, z]
        pos2: 第二个3D点 [x, y, z]

    Returns:
        3D距离
    """
    return np.sqrt(np.sum((pos1 - pos2) ** 2))


def calculate_3d_vector_length(vector: np.ndarray) -> float:
    """
    计算3D向量的长度

    Args:
        vector: 3D向量 [x, y, z]

    Returns:
        向量长度
    """
    return np.linalg.norm(vector)


def normalize_3d_vector(vector: np.ndarray) -> np.ndarray:
    """
    归一化3D向量

    Args:
        vector: 3D向量 [x, y, z]

    Returns:
        归一化后的向量
    """
    length = calculate_3d_vector_length(vector)
    if length == 0:
        return np.array([0, 0, 0])
    return vector / length


def calculate_dot_product_3d(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算3D向量的点积

    Args:
        v1: 第一个3D向量
        v2: 第二个3D向量

    Returns:
        点积结果
    """
    return np.dot(v1, v2)


def calculate_cross_product_3d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    计算3D向量的叉积

    Args:
        v1: 第一个3D向量
        v2: 第二个3D向量

    Returns:
        叉积结果向量
    """
    return np.cross(v1, v2)


def calculate_3d_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两个3D向量之间的夹角 (弧度)

    Args:
        v1: 第一个3D向量
        v2: 第二个3D向量

    Returns:
        夹角 (弧度)
    """
    dot_product = calculate_dot_product_3d(v1, v2)
    magnitudes = calculate_3d_vector_length(v1) * calculate_3d_vector_length(v2)

    if magnitudes == 0:
        return 0.0

    cos_angle = dot_product / magnitudes
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def calculate_3d_plane_equation(points: np.ndarray) -> Tuple[float, float, float, float]:
    """
    计算3D平面方程 Ax + By + Cz + D = 0

    Args:
        points: 平面上的三个点 [3, 3]

    Returns:
        (A, B, C, D) 平面方程系数
    """
    if len(points) != 3:
        raise ValueError("需要三个点来定义平面")

    p1, p2, p3 = points

    # 计算两个向量
    v1 = p2 - p1
    v2 = p3 - p1

    # 计算法向量 (叉积)
    normal = calculate_cross_product_3d(v1, v2)

    # 平面方程: Ax + By + Cz + D = 0
    A, B, C = normal
    D = -np.dot(normal, p1)

    return A, B, C, D


def calculate_distance_point_to_plane(point: np.ndarray,
                                      plane_coeffs: Tuple[float, float, float, float]) -> float:
    """
    计算点到3D平面的距离

    Args:
        point: 3D点 [x, y, z]
        plane_coeffs: 平面方程系数 (A, B, C, D)

    Returns:
        点到平面的距离
    """
    A, B, C, D = plane_coeffs
    x, y, z = point

    numerator = abs(A * x + B * y + C * z + D)
    denominator = math.sqrt(A ** 2 + B ** 2 + C ** 2)

    if denominator == 0:
        return 0.0

    return numerator / denominator


# 测试函数
def test_3d_utils():
    """测试3D工具函数"""
    print("Testing 3D utility functions...")

    try:
        # 测试3D距离计算
        pos1 = np.array([0, 0, 0])
        pos2 = np.array([3, 4, 0])
        dist = calculate_3d_distance(pos1, pos2)
        print(f"3D distance between [0,0,0] and [3,4,0]: {dist:.2f}")

        # 测试向量计算
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        angle = calculate_3d_angle_between_vectors(v1, v2)
        print(f"Angle between [1,0,0] and [0,1,0]: {radians_to_degrees(angle):.1f}°")

        # 测试平面计算
        points = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        A, B, C, D = calculate_3d_plane_equation(points)
        print(f"Plane equation: {A:.1f}x + {B:.1f}y + {C:.1f}z + {D:.1f} = 0")

        print("3D utility function tests completed successfully!")
        return True

    except Exception as e:
        print(f"3D utility function test failed: {e}")
        return False


if __name__ == "__main__":
    test_3d_utils()