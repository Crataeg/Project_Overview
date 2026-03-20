"""
速度优化模块测试
测试VelocityDegradationModel和VelocityOptimizer的功能
"""

import unittest
import numpy as np
import sys
import os

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from velocity import (
    VelocityDegradationModel,
    VelocityOptimizer
)


class TestVelocityDegradationModel(unittest.TestCase):
    """测试速度劣化模型"""

    def setUp(self):
        """测试前准备"""
        self.model = VelocityDegradationModel()
        self.test_velocities = [0, 10, 30, 60, 80, 100]

    def test_doppler_shift_calculation(self):
        """测试多普勒频移计算"""
        # 测试零速度时多普勒频移为0
        doppler_zero = self.model.doppler_shift(0)
        self.assertEqual(doppler_zero, 0)

        # 测试速度增加时多普勒频移增加
        doppler_low = self.model.doppler_shift(10)
        doppler_high = self.model.doppler_shift(50)
        self.assertGreater(doppler_high, doppler_low)

        # 验证多普勒频移公式
        f0_hz = self.model.velocity_params['f0'] * 1e6
        c = self.model.velocity_params['c']
        expected_doppler = (30 * f0_hz) / c
        actual_doppler = self.model.doppler_shift(30)
        self.assertAlmostEqual(actual_doppler, expected_doppler, places=1)

    def test_packet_loss_rate_range(self):
        """测试丢包率在合理范围内"""
        for velocity in self.test_velocities:
            plr = self.model.packet_loss_rate(velocity)
            self.assertGreaterEqual(plr, 0)
            self.assertLessEqual(plr, 1)

    def test_outage_probability_range(self):
        """测试中断概率在合理范围内"""
        for velocity in self.test_velocities:
            outage = self.model.outage_probability(velocity)
            self.assertGreaterEqual(outage, 0)
            self.assertLessEqual(outage, 1)

    def test_handover_failure_range(self):
        """测试切换失败率在合理范围内"""
        for velocity in self.test_velocities:
            handover = self.model.handover_failure(velocity)
            self.assertGreaterEqual(handover, 0)
            self.assertLessEqual(handover, 1)

    def test_degradation_range(self):
        """测试劣化程度在0-1范围内"""
        for velocity in self.test_velocities:
            degradation = self.model.calculate_single_uav_degradation(velocity)
            self.assertGreaterEqual(degradation, 0)
            self.assertLessEqual(degradation, 1)

    def test_degradation_monotonicity(self):
        """测试劣化程度随速度增加而增加"""
        deg_low = self.model.calculate_single_uav_degradation(10)
        deg_high = self.model.calculate_single_uav_degradation(60)
        self.assertGreater(deg_high, deg_low)

    def test_velocity_category_classification(self):
        """测试速度类别分类"""
        test_cases = [
            (0, "低速"),
            (5, "低速"),
            (15, "中速"),
            (25, "中速"),
            (40, "高速"),
            (55, "高速"),
            (70, "超高速"),
            (90, "超高速")
        ]

        for velocity, expected_category in test_cases:
            category = self.model.get_velocity_category(velocity)
            self.assertEqual(category, expected_category)

    def test_weight_sum(self):
        """测试权重总和为1"""
        total_weight = sum(self.model.delta_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)

    def test_zero_velocity(self):
        """测试零速度情况"""
        degradation = self.model.calculate_single_uav_degradation(0)
        # 零速度时应该有基础劣化程度
        self.assertGreaterEqual(degradation, 0)
        self.assertLess(degradation, 0.5)  # 应该相对较低


class TestVelocityOptimizer(unittest.TestCase):
    """测试速度优化器"""

    def setUp(self):
        """测试前准备"""
        self.num_uavs = 6
        self.velocity_range = (0, 80)

        # 创建优化器（使用小种群和少代数以加快测试）
        self.optimizer = VelocityOptimizer(
            num_uavs=self.num_uavs,
            population_size=20,
            max_generations=10,
            velocity_range=self.velocity_range
        )

    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        self.assertEqual(self.optimizer.num_uavs, self.num_uavs)
        self.assertEqual(self.optimizer.population_size, 20)
        self.assertEqual(self.optimizer.max_generations, 10)
        self.assertEqual(self.optimizer.velocity_range, self.velocity_range)
        self.assertIsNotNone(self.optimizer.degradation_model)
        self.assertIsNotNone(self.optimizer.problem)
        self.assertIsNotNone(self.optimizer.algorithm)

    def test_problem_dimension(self):
        """测试问题维度设置"""
        # 决策变量维度应该等于无人机数量
        self.assertEqual(self.optimizer.problem.Dim, self.num_uavs)

        # 变量边界应该正确设置
        self.assertEqual(len(self.optimizer.problem.lb), self.num_uavs)
        self.assertEqual(len(self.optimizer.problem.ub), self.num_uavs)

        # 所有下界应该是最小速度
        self.assertTrue(all(lb == self.velocity_range[0] for lb in self.optimizer.problem.lb))

        # 所有上界应该是最大速度
        self.assertTrue(all(ub == self.velocity_range[1] for ub in self.optimizer.problem.ub))

    def test_optimization_run(self):
        """测试优化运行"""
        result = self.optimizer.run_optimization()

        # 检查结果包含必要的键
        required_keys = [
            'best_velocities',
            'best_fitness',
            'uav_degradations',
            'statistics',
            'optimization_parameters'
        ]

        for key in required_keys:
            self.assertIn(key, result)

        # 检查最优速度组合
        best_velocities = result['best_velocities']
        self.assertEqual(len(best_velocities), self.num_uavs)

        # 检查所有速度在合理范围内
        for velocity in best_velocities:
            self.assertGreaterEqual(velocity, self.velocity_range[0])
            self.assertLessEqual(velocity, self.velocity_range[1])

        # 检查适应度值在合理范围内
        fitness = result['best_fitness']
        self.assertGreaterEqual(fitness, 0)
        self.assertLessEqual(fitness, 1)

        # 检查统计信息
        stats = result['statistics']
        self.assertIn('average_velocity', stats)
        self.assertIn('max_velocity', stats)
        self.assertIn('min_velocity', stats)
        self.assertIn('velocity_range', stats)

        self.assertGreaterEqual(stats['average_velocity'], self.velocity_range[0])
        self.assertLessEqual(stats['average_velocity'], self.velocity_range[1])
        self.assertGreaterEqual(stats['min_velocity'], self.velocity_range[0])
        self.assertLessEqual(stats['max_velocity'], self.velocity_range[1])

        # 检查每个无人机的劣化信息
        self.assertEqual(len(result['uav_degradations']), self.num_uavs)
        for uav_info in result['uav_degradations']:
            required_uav_keys = ['uav_id', 'velocity', 'velocity_category', 'degradation', 'doppler_shift']
            for key in required_uav_keys:
                self.assertIn(key, uav_info)

            self.assertGreaterEqual(uav_info['velocity'], self.velocity_range[0])
            self.assertLessEqual(uav_info['velocity'], self.velocity_range[1])
            self.assertIn(uav_info['velocity_category'], ["低速", "中速", "高速", "超高速"])
            self.assertGreaterEqual(uav_info['degradation'], 0)
            self.assertLessEqual(uav_info['degradation'], 1)
            self.assertGreaterEqual(uav_info['doppler_shift'], 0)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_single_uav_optimization(self):
        """测试单无人机优化"""
        optimizer = VelocityOptimizer(
            num_uavs=1,
            population_size=10,
            max_generations=5,
            velocity_range=(0, 50)
        )
        result = optimizer.run_optimization()

        # 单无人机情况下，最优速度应该接近最大速度以获得最大劣化
        best_velocity = result['best_velocities'][0]
        self.assertGreater(best_velocity, 25)  # 应该偏向较高速度

    def test_minimal_velocity(self):
        """测试最小速度"""
        model = VelocityDegradationModel()
        degradation = model.calculate_single_uav_degradation(0)

        # 最小速度应该产生较低的劣化程度
        self.assertLess(degradation, 0.5)

    def test_maximal_velocity(self):
        """测试最大速度"""
        model = VelocityDegradationModel()
        degradation = model.calculate_single_uav_degradation(100)

        # 最大速度应该产生较高的劣化程度
        self.assertGreater(degradation, 0.3)

    def test_narrow_velocity_range(self):
        """测试狭窄速度范围"""
        optimizer = VelocityOptimizer(
            num_uavs=4,
            population_size=15,
            max_generations=5,
            velocity_range=(20, 30)  # 狭窄范围
        )
        result = optimizer.run_optimization()

        stats = result['statistics']
        self.assertGreaterEqual(stats['min_velocity'], 20)
        self.assertLessEqual(stats['max_velocity'], 30)


class TestIntegration(unittest.TestCase):
    """测试集成功能"""

    def test_summary_generation(self):
        """测试结果摘要生成"""
        optimizer = VelocityOptimizer(
            num_uavs=3,
            population_size=10,
            max_generations=5
        )
        result = optimizer.run_optimization()

        summary = optimizer.get_optimization_summary()

        # 检查摘要包含关键信息
        self.assertIn("无人机数量", summary)
        self.assertIn("平均速度", summary)
        self.assertIn("速度范围", summary)
        self.assertIn("平均通信劣化程度", summary)
        self.assertIn("速度类别统计", summary)

    def test_plotting_functions(self):
        """测试绘图功能（不实际显示）"""
        optimizer = VelocityOptimizer(
            num_uavs=4,
            population_size=10,
            max_generations=5
        )
        result = optimizer.run_optimization()

        # 检查绘图函数是否能正常调用而不出错
        try:
            # 使用非交互式后端避免显示图表
            import matplotlib
            matplotlib.use('Agg')

            optimizer.plot_optimization_results()
            optimizer.plot_velocity_degradation_analysis()
            plot_success = True
        except Exception as e:
            plot_success = False
            print(f"绘图测试中出现错误: {e}")

        self.assertTrue(plot_success)

    def test_velocity_distribution(self):
        """测试速度分布合理性"""
        optimizer = VelocityOptimizer(
            num_uavs=10,
            population_size=30,
            max_generations=10
        )
        result = optimizer.run_optimization()

        velocities = result['best_velocities']

        # 检查速度分布不是所有都相同（除非确实最优）
        unique_velocities = len(set(round(v, 1) for v in velocities))
        self.assertGreater(unique_velocities, 1)  # 应该有不同的速度值


class TestPerformance(unittest.TestCase):
    """测试性能相关功能"""

    def test_degradation_calculation_performance(self):
        """测试劣化程度计算性能"""
        import time

        model = VelocityDegradationModel()

        start_time = time.time()
        for _ in range(1000):
            _ = model.calculate_single_uav_degradation(30)
        end_time = time.time()

        computation_time = end_time - start_time
        self.assertLess(computation_time, 1.0)  # 应该在1秒内完成1000次计算

    def test_optimization_performance(self):
        """测试优化性能"""
        import time

        start_time = time.time()

        optimizer = VelocityOptimizer(
            num_uavs=20,
            population_size=50,
            max_generations=20
        )

        result = optimizer.run_optimization()
        end_time = time.time()

        computation_time = end_time - start_time
        # 检查优化在合理时间内完成
        self.assertLess(computation_time, 30.0)  # 30秒内完成


class TestModelConsistency(unittest.TestCase):
    """测试模型一致性"""

    def test_parameter_consistency(self):
        """测试参数一致性"""
        model = VelocityDegradationModel()

        # 检查参数范围一致性
        self.assertGreater(model.velocity_params['v_max'], 0)
        self.assertGreater(model.velocity_params['f0'], 0)
        self.assertGreater(model.velocity_params['c'], 0)

        # 检查系数均为正数
        for coeff_value in model.coefficients.values():
            self.assertGreater(coeff_value, 0)

    def test_function_consistency(self):
        """测试函数一致性"""
        model = VelocityDegradationModel()

        # 测试所有函数在边界值的行为
        test_velocities = [0, 50, 100]

        for velocity in test_velocities:
            # 所有函数都应该能够处理这些速度值而不出错
            doppler = model.doppler_shift(velocity)
            plr = model.packet_loss_rate(velocity)
            outage = model.outage_probability(velocity)
            degradation = model.calculate_single_uav_degradation(velocity)

            # 检查返回值类型
            self.assertIsInstance(doppler, float)
            self.assertIsInstance(plr, float)
            self.assertIsInstance(outage, float)
            self.assertIsInstance(degradation, float)


def run_performance_test():
    """运行性能测试（非单元测试）"""
    print("运行速度优化性能测试...")

    # 测试大规模场景
    num_uavs = 50

    import time
    start_time = time.time()

    optimizer = VelocityOptimizer(
        num_uavs=num_uavs,
        population_size=60,
        max_generations=30
    )

    result = optimizer.run_optimization()
    end_time = time.time()

    print(f"性能测试完成:")
    print(f"  无人机数量: {num_uavs}")
    print(f"  运行时间: {end_time - start_time:.2f}秒")
    print(f"  最优劣化程度: {result['best_fitness']:.4f}")
    print(f"  平均速度: {result['statistics']['average_velocity']:.2f} m/s")
    print(f"  速度范围: {result['statistics']['min_velocity']:.2f} - {result['statistics']['max_velocity']:.2f} m/s")

    return result


if __name__ == '__main__':
    # 运行单元测试
    print("运行速度优化模块单元测试...")
    unittest.main(verbosity=2)

    # 可选：运行性能测试
    # print("\n" + "="*50)
    # run_performance_test()

    # 可选：运行快速测试
    # from velocity import quick_test
    # quick_test()