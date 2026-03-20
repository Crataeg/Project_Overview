"""
遮挡优化中间件
提供遮挡相关的通信劣化程度计算和遮挡物位置优化功能（支持三维）
"""

from .obstruction import (
    ObstructionDegradationModel,
    ObstructionEffectModel3D,
    ObstructionOptimizationProblem3D,
    ObstructionOptimizer3D
)

__all__ = [
    'ObstructionDegradationModel',
    'ObstructionEffectModel3D',
    'ObstructionOptimizationProblem3D',
    'ObstructionOptimizer3D'
]

__version__ = '2.0.0'
__author__ = '无人机通信优化系统'

# 包级别的默认配置
DEFAULT_PARAMETERS = {
    'max_obstruction': 1.0,
    'obstruction_outage_threshold': 0.8,
    'horizontal_range': 300.0,
    'vertical_range': 150.0,
    'default_obstruction_height': 50.0
}


def get_module_info():
    """返回模块信息"""
    return {
        'name': '三维遮挡优化中间件',
        'version': __version__,
        'description': '处理三维遮挡因素对无人机通信劣化程度的影响',
        'author': __author__,
        'main_classes': __all__
    }


def quick_test():
    """快速测试函数"""
    print("三维遮挡优化模块快速测试...")
    model = ObstructionDegradationModel()

    # 测试不同遮挡程度的劣化程度
    test_levels = [0.1, 0.3, 0.6, 0.9]
    print("遮挡程度\t劣化程度")
    print("-" * 20)
    for level in test_levels:
        degradation = model.calculate_single_uav_degradation(level)
        print(f"{level}\t\t{degradation:.4f}")

    return True