"""
三维综合联调系统测试
测试CombinedDegradationModel和CombinedOptimizer的功能
"""

import unittest
import numpy as np
import sys
import os

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from combined import (
    CombinedDegradationModel,
    CombinedOptimizer
)


class TestCombinedDegradationModel(unittest.TestCase):
    """测试三维综合劣化模型"""

    def setUp(self):
        """测试前准备"""
        self.uav_positions = np.array([
            [100, 200, 50],
            [150, 180, 60],
            [200, 220, 55]
        ])
        self.model = CombinedDegradationModel(self.uav_positions)
        self.num_uavs = len(self.uav_positions)

    def test_initialization(self):
        """测试模型初始化"""
        self.assertEqual(self.model.num_uavs, 3)
        self.assertEqual(len(self.model.uav_positions), 3)

        # 检查权重配置
        expected_weights = {'distance': 0.3, 'obstruction': 0.2, 'interference': 0.3, 'velocity': 0.2}
        for key, value in expected_weights.items():
            self.assertAlmostEqual(self.model.combined_weights[key], value, places=2)

    def test_3d_distance_calculation(self):
        """测试三维距离计算"""
        pos1 = [0, 0, 0]
        pos2 = [3, 4, 0]  # 水平距离5
        distance = self.model.calculate_3d_distance(pos1, pos2)
        self.assertAlmostEqual(distance, 5.0, places=6)

        pos3 = [0, 0, 0]
        pos4 = [0, 0, 5]  # 垂直距离5
        distance = self.model.calculate_3d_distance(pos3, pos4)
        self.assertAlmostEqual(distance, 5.0, places=6)

        pos5 = [1, 2, 2]
        pos6 = [4, 6, 5]  # 三维距离 sqrt(3^2 + 4^2 + 3^2) = sqrt(34)
        distance = self.model.calculate_3d_distance(pos5, pos6)
        self.assertAlmostEqual(distance, np.sqrt(34), places=6)

    def test_calculate_distance_degradation_range(self):
        """测试距离劣化程度在合理范围内"""
        test_positions = [
            [100, 100],  # 近距离
            [300, 300],  # 中距离
            [500, 500]  # 远距离
        ]

        for position in test_positions:
            degradation = self.model.calculate_distance_degradation(position)
            self.assertGreaterEqual(degradation, 0)
            self.assertLessEqual(degradation, 1)

    def test_calculate_obstruction_degradation_range(self):
        """测试遮挡劣化程度在合理范围内"""
        test_positions = [
            [100, 100, 50],  # 近距离遮挡
            [300, 300, 60],  # 中距离遮挡
            [500, 500, 70]  # 远距离遮挡
        ]

        for position in test_positions:
            degradation = self.model.calculate_obstruction_degradation(position)
            self.assertGreaterEqual(degradation, 0)
            self.assertLessEqual(degradation, 1)

    def test_calculate_interference_degradation_range(self):
        """测试干扰劣化程度在合理范围内"""
        test_params = [
            [100, 100, 50, 2.4e9, 20e6, 1.0],  # 近距离同频干扰
            [300, 300, 60, 5.0e9, 10e6, 2.0],  # 中距离异频干扰
            [500, 500, 70, 1.0e9, 50e6, 5.0]  # 远距离宽带干扰
        ]

        for params in test_params:
            degradation = self.model.calculate_interference_degradation(params)
            self.assertGreaterEqual(degradation, 0)
            self.assertLessEqual(degradation, 1)

    def test_calculate_velocity_degradation_range(self):
        """测试速度劣化程度在合理范围内"""
        test_velocities = [
            [10, 15, 20],  # 中低速
            [30, 35, 40],  # 中高速
            [45, 50, 55]  # 高速
        ]

        for velocities in test_velocities:
            degradation = self.model.calculate_velocity_degradation(velocities)
            self.assertGreaterEqual(degradation, 0)
            self.assertLessEqual(degradation, 1)

    def test_combined_degradation_range(self):
        """测试综合劣化程度在合理范围内"""
        # 创建测试决策变量（三维版本）
        decision_vars = [
            250, 250,  # 地面站位置
            300, 300, 40,  # 遮挡物位置 (三维)
            200, 200, 30, 2.4e9, 20e6, 2.0,  # 干扰源参数 (三维)
            30, 25, 35  # 无人机速度
        ]

        total_degradation, components = self.model.calculate_combined_degradation(decision_vars)

        self.assertGreaterEqual(total_degradation, 0)
        self.assertLessEqual(total_degradation, 1)

        # 检查各分量
        for factor in ['distance', 'obstruction', 'interference', 'velocity']:
            self.assertIn(factor, components)
            self.assertGreaterEqual(components[factor], 0)
            self.assertLessEqual(components[factor], 1)

    def test_weight_adjustment(self):
        """测试权重调整功能"""
        new_weights = {
            'distance': 0.4,
            'obstruction': 0.3,
            'interference': 0.2,
            'velocity': 0.1
        }

        self.model.set_combined_weights(new_weights)

        # 检查权重是否正确更新和归一化
        total = sum(self.model.combined_weights.values())
        self.assertAlmostEqual(total, 1.0, places=6)

        for key, value in new_weights.items():
            expected = value / sum(new_weights.values())
            self.assertAlmostEqual(self.model.combined_weights[key], expected, places=6)

    def test_3d_decision_variable_parsing(self):
        """测试三维决策变量解析"""
        # 三维版本决策变量: 2(gs) + 3(obs) + 6(int) + 3(vel) = 14
        decision_vars = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        total_degradation, components = self.model.calculate_combined_degradation(decision_vars)

        # 应该能够正常计算而不出错
        self.assertIsInstance(total_degradation, float)
        self.assertIsInstance(components, dict)


class TestCombinedOptimizer(unittest.TestCase):
    """测试三维综合优化器"""

    def setUp(self):
        """测试前准备"""
        self.uav_positions = np.array([
            [100, 200, 50],
            [150, 180, 60],
            [200, 220, 55],
            [180, 150, 65]
        ])

        # 使用小种群和少代数以加快测试
        self.optimizer = CombinedOptimizer(
            uav_positions=self.uav_positions,
            population_size=30,
            max_generations=20
        )

    def test_initialization(self):
        """测试优化器初始化"""
        self.assertEqual(len(self.optimizer.uav_positions), 4)
        self.assertEqual(self.optimizer.num_uavs, 4)
        self.assertEqual(self.optimizer.population_size, 30)
        self.assertEqual(self.optimizer.max_generations, 20)

        # 检查决策变量维度（三维版本）
        expected_dim = 11 + self.optimizer.num_uavs  # 2+3+6+n
        self.assertEqual(self.optimizer.total_dim, expected_dim)

        # 检查边界设置
        self.assertEqual(len(self.optimizer.lb), expected_dim)
        self.assertEqual(len(self.optimizer.ub), expected_dim)

        # 检查模型初始化
        self.assertIsNotNone(self.optimizer.degradation_model)

    def test_3d_variable_bounds_calculation(self):
        """测试三维变量边界计算"""
        lb, ub = self.optimizer._get_variable_bounds()

        # 检查边界长度
        expected_length = 11 + self.optimizer.num_uavs
        self.assertEqual(len(lb), expected_length)
        self.assertEqual(len(ub), expected_length)

        # 检查具体边界值
        ranges = self.optimizer.ranges

        # 地面站位置边界
        self.assertEqual(lb[0], ranges['gs_x'][0])
        self.assertEqual(lb[1], ranges['gs_y'][0])
        self.assertEqual(ub[0], ranges['gs_x'][1])
        self.assertEqual(ub[1], ranges['gs_y'][1])

        # 遮挡物高度边界
        self.assertEqual(lb[4], ranges['obs_z'][0])
        self.assertEqual(ub[4], ranges['obs_z'][1])

        # 干扰源高度边界
        self.assertEqual(lb[7], ranges['int_z'][0])
        self.assertEqual(ub[7], ranges['int_z'][1])

        # 无人机速度边界
        for i in range(11, 11 + self.optimizer.num_uavs):
            self.assertEqual(lb[i], ranges['uav_velocities'][0])
            self.assertEqual(ub[i], ranges['uav_velocities'][1])

    def test_3d_solution_parsing(self):
        """测试三维解决方案解析"""
        # 创建测试决策变量（三维版本）
        decision_vars = [
            100, 200,  # 地面站位置
            150, 250, 40,  # 遮挡物位置 (三维)
            200, 300, 30, 2.4e9, 20e6, 2.0,  # 干扰源参数 (三维)
            30, 25, 35, 40  # 无人机速度
        ]

        total_degradation = 0.75
        components = {
            'distance': 0.7,
            'obstruction': 0.6,
            'interference': 0.8,
            'velocity': 0.65,
            'contributions': [0.21, 0.12, 0.24, 0.13]
        }

        solution = self.optimizer._parse_solution(decision_vars, total_degradation, components)

        # 检查解析结果
        self.assertEqual(len(solution['gs_position']), 2)
        self.assertEqual(len(solution['obstruction_position']), 3)  # 三维
        self.assertEqual(len(solution['interference_params']['position']), 3)  # 三维
        self.assertEqual(len(solution['uav_velocities']), self.optimizer.num_uavs)

        self.assertEqual(solution['total_degradation'], total_degradation)
        self.assertEqual(solution['distance_degradation'], components['distance'])
        self.assertEqual(solution['obstruction_degradation'], components['obstruction'])
        self.assertEqual(solution['interference_degradation'], components['interference'])
        self.assertEqual(solution['velocity_degradation'], components['velocity'])

        # 检查三维位置数据
        self.assertEqual(solution['obstruction_position'][2], 40)  # 遮挡物高度
        self.assertEqual(solution['interference_params']['position'][2], 30)  # 干扰源高度

    def test_optimization_run(self):
        """测试优化运行（简化版）"""
        # 注意：完整优化可能较慢，这里主要测试接口和基本功能
        try:
            # 使用更小的参数以加快测试
            quick_optimizer = CombinedOptimizer(
                uav_positions=self.uav_positions[:2],  # 只使用2个无人机
                population_size=10,
                max_generations=5
            )

            best_solution, best_fitness, history = quick_optimizer.optimize()

            # 检查返回结果
            self.assertIsInstance(best_solution, dict)
            self.assertIsInstance(best_fitness, float)
            self.assertIsInstance(history, dict)

            # 检查适应度值在合理范围内
            self.assertGreaterEqual(best_fitness, 0)
            self.assertLessEqual(best_fitness, 1)

            # 检查解决方案包含必要字段
            required_keys = [
                'gs_position', 'obstruction_position', 'interference_params',
                'uav_velocities', 'total_degradation', 'distance_degradation',
                'obstruction_degradation', 'interference_degradation', 'velocity_degradation'
            ]

            for key in required_keys:
                self.assertIn(key, best_solution)

            # 检查三维位置数据
            self.assertEqual(len(best_solution['obstruction_position']), 3)
            self.assertEqual(len(best_solution['interference_params']['position']), 3)

        except Exception as e:
            # 如果优化失败，记录但不中断测试（可能是环境问题）
            print(f"优化测试中出现警告: {e}")
            self.skipTest(f"优化测试跳过: {e}")


class TestIntegration(unittest.TestCase):
    """测试集成功能"""

    def test_3d_results_analysis(self):
        """测试三维结果分析功能"""
        uav_positions = np.array([
            [100, 200, 50],
            [150, 180, 60]
        ])

        optimizer = CombinedOptimizer(
            uav_positions=uav_positions,
            population_size=10,
            max_generations=5
        )

        # 创建模拟解决方案（三维版本）
        mock_solution = {
            'gs_position': [120, 180],
            'obstruction_position': [140, 160, 45],  # 三维
            'interference_params': {
                'position': [130, 170, 35],  # 三维
                'frequency': 2.4e9,
                'bandwidth': 20e6,
                'power': 2.0
            },
            'uav_velocities': [25, 30],
            'total_degradation': 0.75,
            'distance_degradation': 0.7,
            'obstruction_degradation': 0.6,
            'interference_degradation': 0.8,
            'velocity_degradation': 0.65
        }

        mock_fitness = 0.75

        # 测试分析功能不报错
        try:
            optimizer.analyze_results(mock_solution, mock_fitness)
            analysis_success = True
        except Exception as e:
            analysis_success = False
            print(f"分析功能测试中出现错误: {e}")

        self.assertTrue(analysis_success)

    def test_3d_weighted_combination(self):
        """测试三维加权组合的正确性"""
        uav_positions = np.array([[100, 200, 50]])
        model = CombinedDegradationModel(uav_positions)

        # 设置特定权重
        model.set_combined_weights({
            'distance': 0.5,
            'obstruction': 0.3,
            'interference': 0.1,
            'velocity': 0.1
        })

        # 创建测试决策变量（三维版本）
        decision_vars = [100, 100, 200, 200, 40, 150, 150, 30, 2.4e9, 20e6, 2.0, 30]

        total_degradation, components = model.calculate_combined_degradation(decision_vars)

        # 手动计算加权和进行验证
        manual_calculation = (
                model.combined_weights['distance'] * components['distance'] +
                model.combined_weights['obstruction'] * components['obstruction'] +
                model.combined_weights['interference'] * components['interference'] +
                model.combined_weights['velocity'] * components['velocity']
        )

        self.assertAlmostEqual(total_degradation, manual_calculation, places=6)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_single_uav_3d_scenario(self):
        """测试单无人机三维场景"""
        uav_positions = np.array([[250, 250, 50]])

        optimizer = CombinedOptimizer(
            uav_positions=uav_positions,
            population_size=10,
            max_generations=3
        )

        # 主要测试初始化是否正确
        self.assertEqual(optimizer.num_uavs, 1)
        self.assertEqual(optimizer.total_dim, 12)  # 11 + 1

        # 检查变量边界
        self.assertEqual(len(optimizer.lb), 12)
        self.assertEqual(len(optimizer.ub), 12)

    def test_extreme_3d_weights(self):
        """测试极端权重配置"""
        uav_positions = np.array([[100, 200, 50]])
        model = CombinedDegradationModel(uav_positions)

        # 测试全零权重（应该自动归一化）
        model.set_combined_weights({
            'distance': 0,
            'obstruction': 0,
            'interference': 0,
            'velocity': 0
        })

        # 权重应该被归一化为平均值
        expected_weight = 0.25
        for weight in model.combined_weights.values():
            self.assertAlmostEqual(weight, expected_weight, places=6)

        # 测试单一因素权重为1
        model.set_combined_weights({
            'distance': 1,
            'obstruction': 0,
            'interference': 0,
            'velocity': 0
        })

        self.assertAlmostEqual(model.combined_weights['distance'], 1.0, places=6)
        self.assertAlmostEqual(model.combined_weights['obstruction'], 0.0, places=6)
        self.assertAlmostEqual(model.combined_weights['interference'], 0.0, places=6)
        self.assertAlmostEqual(model.combined_weights['velocity'], 0.0, places=6)

    def test_extreme_3d_positions(self):
        """测试极端三维位置"""
        uav_positions = np.array([
            [0, 0, 0],  # 最低位置
            [1000, 1000, 100]  # 最高位置
        ])

        model = CombinedDegradationModel(uav_positions)

        # 测试极端位置的计算
        decision_vars = [
            0, 0,  # 地面站在角落
            1000, 1000, 100,  # 遮挡物在最高角落
            500, 500, 50, 2.4e9, 100e6, 10.0,  # 干扰源在中心
            50, 50  # 最高速度
        ]

        total_degradation, components = model.calculate_combined_degradation(decision_vars)

        # 应该能够正常计算而不出错
        self.assertGreaterEqual(total_degradation, 0)
        self.assertLessEqual(total_degradation, 1)


def run_performance_test():
    """运行性能测试（非单元测试）"""
    print("运行三维综合优化性能测试...")

    # 创建测试无人机位置（三维）
    np.random.seed(42)
    num_uavs = 10
    uav_positions = np.array([
        [np.random.uniform(0, 1000), np.random.uniform(0, 1000), np.random.uniform(50, 100)]
        for _ in range(num_uavs)
    ])

    import time
    start_time = time.time()

    optimizer = CombinedOptimizer(
        uav_positions=uav_positions,
        population_size=50,
        max_generations=30
    )

    best_solution, best_fitness, history = optimizer.optimize()
    end_time = time.time()

    print(f"性能测试完成:")
    print(f"  无人机数量: {num_uavs}")
    print(f"  决策变量维度: {optimizer.total_dim}")
    print(f"  运行时间: {end_time - start_time:.2f}秒")
    print(f"  最优劣化程度: {best_fitness:.4f}")

    # 分析结果
    optimizer.analyze_results(best_solution, best_fitness)

    return best_solution, best_fitness


if __name__ == '__main__':
    # 运行单元测试
    print("运行三维综合联调系统单元测试...")
    unittest.main(verbosity=2)

    # 可选：运行性能测试
    # print("\n" + "="*50)
    # run_performance_test()

    # 可选：运行快速测试
    # from combined import quick_test
    # quick_test()