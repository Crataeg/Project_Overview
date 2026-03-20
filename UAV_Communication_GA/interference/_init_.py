"""
无人机通信干扰优化系统
基于遗传算法优化干扰源参数，考虑距离、频率和高度因素对通信质量的影响
"""

__version__ = "1.0.0"
__author__ = "UAV Communication GA Team"
__email__ = "contact@example.com"

# 导入主要类以便可以直接从包中导入
from .interference import (
    InterferenceDegradationModel,
    InterferenceOptimizationProblem,
    InterferenceOptimizer
)

# 定义包的公共API
__all__ = [
    "InterferenceDegradationModel",
    "InterferenceOptimizationProblem",
    "InterferenceOptimizer",
    "main"
]

# 包级别的配置
DEFAULT_CONFIG = {
    "max_uav_height": 500,
    "optimal_uav_height": 150,
    "jammer_max_height": 50,
    "frequency_range": (100, 6000),
    "bandwidth_range": (1, 100),
    "power_range": (0.1, 10.0),
    "search_area_padding": 0.5
}

def get_version():
    """返回包版本信息"""
    return __version__

def get_default_config():
    """返回默认配置"""
    return DEFAULT_CONFIG.copy()

def get_available_bands():
    """返回可用的干扰频段"""
    model = InterferenceDegradationModel()
    return model.interference_bands.copy()

# 当导入包时显示欢迎信息
print(f"初始化 UAV 干扰优化系统 v{__version__}")