"""
速度优化中间件
提供速度相关的通信劣化程度计算和无人机速度优化功能
"""

from .velocity import (
    VelocityDegradationModel,
    VelocityOptimizationProblem,
    VelocityOptimizer
)

__all__ = [
    'VelocityDegradationModel',
    'VelocityOptimizationProblem',
    'VelocityOptimizer'
]

__version__ = '1.0.0'
__author__ = '无人机通信优化系统'

# 包级别的默认配置
DEFAULT_PARAMETERS = {
    'max_velocity': 100,  # m/s
    'carrier_frequency': 2400,  # MHz
    'velocity_categories': {
        'low': (0, 10),
        'medium': (10, 30),
        'high': (30, 60),
        'ultra_high': (60, 100)
    }
}


def get_module_info():
    """返回模块信息"""
    return {
        'name': '速度优化中间件',
        'version': __version__,
        'description': '处理速度因素对无人机通信劣化程度的影响',
        'author': __author__,
        'main_classes': __all__
    }


def quick_test():
    """快速测试函数"""
    print("速度优化模块快速测试...")
    model = VelocityDegradationModel()

    # 测试不同速度的劣化程度
    test_velocities = [0, 10, 30, 60, 80]
    print("速度(m/s)\t多普勒频移(Hz)\t劣化程度\t速度类别")
    print("-" * 60)
    for velocity in test_velocities:
        doppler = model.doppler_shift(velocity)
        degradation = model.calculate_single_uav_degradation(velocity)
        category = model.get_velocity_category(velocity)
        print(f"{velocity}\t\t{doppler:.1f}\t\t{degradation:.4f}\t\t{category}")

    return True