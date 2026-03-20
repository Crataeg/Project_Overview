"""
综合联调系统
组合距离、遮挡、干扰、速度四个中间件进行全局优化（三维版本）
总通信劣化程度 = W₁ × D₁(d) + W₂ × D₂(o) + W₃ × D₃(i) + W₄ × D₄(v)
"""
import numpy as np

from .combined import (
    CombinedDegradationModel,
    CombinedOptimizer
)

__all__ = [
    'CombinedDegradationModel',
    'CombinedOptimizer'
]

__version__ = '2.0.0'
__author__ = '无人机通信优化系统'

# 包级别的默认配置
DEFAULT_PARAMETERS = {
    'default_weights': {
        'distance': 0.3,
        'obstruction': 0.2,
        'interference': 0.3,
        'velocity': 0.2
    },
    'variable_ranges': {
        'gs_position': [0, 1000],
        'obstruction_position': [0, 1000],
        'obstruction_height': [0, 100],
        'interference_position': [0, 1000],
        'interference_height': [0, 100],
        'interference_frequency': [100e6, 6000e6],
        'interference_bandwidth': [1e6, 100e6],
        'interference_power': [0.1, 10],
        'uav_velocities': [0, 50]
    }
}


def get_module_info():
    """返回模块信息"""
    return {
        'name': '三维综合联调系统',
        'version': __version__,
        'description': '组合四个因素进行无人机通信劣化程度全局优化（支持三维空间）',
        'author': __author__,
        'main_classes': __all__,
        'formula': '总通信劣化程度 = W₁ × D₁(d) + W₂ × D₂(o) + W₃ × D₃(i) + W₄ × D₄(v)',
        'features': ['三维距离计算', '三维遮挡模型', '三维干扰模型', '高度感知优化']
    }


def quick_test():
    """快速测试函数"""
    print("三维综合联调系统快速测试...")

    # 创建测试无人机位置（三维）
    uav_positions = np.array([
        [100, 200, 50],
        [150, 180, 60],
        [200, 220, 55]
    ])

    model = CombinedDegradationModel(uav_positions)

    # 测试综合劣化计算
    test_decision_vars = [
        250, 250,  # 地面站位置
        300, 300, 40,  # 遮挡物位置 (三维)
        200, 200, 30, 2.4e9, 20e6, 2.0,  # 干扰源参数 (三维)
        30, 25, 35  # 无人机速度
    ]

    total_degradation, components = model.calculate_combined_degradation(test_decision_vars)

    print("测试决策变量:")
    print(f"  地面站位置: ({test_decision_vars[0]}, {test_decision_vars[1]})")
    print(f"  遮挡物位置: ({test_decision_vars[2]}, {test_decision_vars[3]}, {test_decision_vars[4]})")
    print(f"  干扰源参数: 位置({test_decision_vars[5]}, {test_decision_vars[6]}, {test_decision_vars[7]}), "
          f"频率{test_decision_vars[8] / 1e6:.1f}MHz, 带宽{test_decision_vars[9] / 1e6:.1f}MHz, 功率{test_decision_vars[10]:.1f}W")
    print(f"  无人机速度: {test_decision_vars[11:]}")

    print(f"\n综合劣化程度: {total_degradation:.4f}")
    print("各分量劣化程度:")
    for factor, value in components.items():
        if factor != 'contributions':
            print(f"  {factor}: {value:.4f}")

    return True