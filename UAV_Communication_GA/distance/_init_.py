"""
无人机通信劣化程度优化系统
基于遗传算法优化地面站位置，考虑距离和高度因素对通信质量的影响
"""

__version__ = "1.0.0"
__author__ = "UAV Communication GA Team"
__email__ = "contact@example.com"

# 导入主要类以便可以直接从包中导入
from .distance import (
    DistanceDegradationModel,
    GroundStationOptimizationProblem,
    DistanceOptimizer
)

# 定义包的公共API
__all__ = [
    "DistanceDegradationModel",
    "GroundStationOptimizationProblem",
    "DistanceOptimizer",
    "main"
]

# 包级别的配置
DEFAULT_CONFIG = {
    "max_uav_height": 500,
    "optimal_uav_height": 150,
    "gs_max_height": 50,
    "search_area_padding": 0.5
}

def get_version():
    """返回包版本信息"""
    return __version__

def get_default_config():
    """返回默认配置"""
    return DEFAULT_CONFIG.copy()

# 当导入包时显示欢迎信息
print(f"初始化 UAV 通信优化系统 v{__version__}")