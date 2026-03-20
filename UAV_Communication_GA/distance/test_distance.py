"""
无人机通信劣化程度优化系统 - 单元测试
测试距离和高度模型、优化算法的正确性
"""

import unittest
import numpy as np
import sys
import os

# 添加当前目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from distance import DistanceDegradationModel, DistanceOptimizer


class TestDistanceDegradationModel(unittest.TestCase):
    """测试距离劣化程度计算模型"""

    def setUp(self):
        """测试前初始化"""
        self.model = DistanceDegradationModel()

    def test_path_loss(self):
        """测试路径损耗计算"""
        # 测试参考距离处的路径损耗
        pl_at_d0 = self.model.path_loss(self.model.distance_params['d0'])
        self.assertAlmostEqual(pl_at_d0, self.model.distance_params['PL0'], delta=0.1)

        # 测试距离增加时路径损耗增加
        pl_100 = self.model.path_loss(100)
        pl_200 = self.model.path_loss(200)
        self.assertGreater(pl_200, pl_100)

    def test_height_penalty(self):
        """测试高度惩罚函数"""
        # 测试最优高度附近的惩罚
        penalty_optimal = self.model.height_penalty(self.model.distance_params['h_optimal'])
        self.assertLess(penalty_optimal, 0.1)

        # 测试高度偏离时的惩罚增加
        penalty_low = self.model.height_penalty(50)
        penalty_high = self.model.height_penalty(400)
        self.assertGreater(penalty_high, penalty_low)

        # 测试惩罚值在0-1范围内
        self.assertGreaterEqual(penalty_optimal, 0)
        self.assertLessEqual(penalty_optimal, 1)

    def test_packet_loss_rate(self):
        """测试丢包率计算"""
        # 测试基础丢包率
        plr_at_d0 = self.model.packet_loss_rate(0)
        self.assertAlmostEqual(plr_at_d0, self.model.distance_params['PLR0'], delta=0.01)

        # 测试丢包率在0-1范围内
        plr = self.model.packet_loss_rate(1000)
        self.assertGreaterEqual(plr, 0)
        self.assertLessEqual(plr, 1)

    def test_outage_probability(self):
        """测试中断概率计算"""
        # 测试中断概率在0-1范围内
        p_out = self.model.outage_probability(500)
        self.assertGreaterEqual(p_out, 0)
        self.assertLessEqual(p_out, 1)

        # 测试距离增加时中断概率增加
        p_out_close = self.model.outage_probability(100)
        p_out_far = self.model.outage_probability(1000)
        self.assertGreater(p_out_far, p_out_close)

    def test_degradation_calculation(self):
        """测试劣化程度计算"""
        # 测试劣化程度在0-1范围内
        degradation = self.model.calculate_single_uav_degradation(500, 200)
        self.assertGreaterEqual(degradation, 0)
        self.assertLessEqual(degradation, 1)

        # 测试距离增加时劣化程度增加
        deg_close = self.model.calculate_single_uav_degradation(100, 150)
        deg_far = self.model.calculate_single_uav_degradation(1500, 150)
        self.assertGreater(deg_far, deg_close)

        # 测试高度偏离最优值时劣化程度增加
        deg_optimal = self.model.calculate_single_uav_degradation(500, 150)
        deg_high = self.model.calculate_single_uav_degradation(500, 400)
        self.assertGreater(deg_high, deg_optimal)


class TestDistanceOptimizer(unittest.TestCase):
    """测试距离优化器"""

    def setUp(self):
        """测试前初始化"""
        # 创建测试用的无人机位置
        self.uav_positions = [
            (100, 100, 150),
            (200, 200, 200),
            (300, 100, 100),
            (100, 300, 250)
        ]

        # 使用较小的种群和代数进行快速测试
        self.optimizer = DistanceOptimizer(
            uav_positions=self.uav_positions,
            population_size=10,
            max_generations=5
        )

    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        self.assertEqual(len(self.optimizer.uav_positions), 4)
        self.assertEqual(self.optimizer.population_size, 10)
        self.assertEqual(self.optimizer.max_generations, 5)

        # 测试问题定义
        self.assertEqual(self.optimizer.problem.Dim, 3)  # 三维优化问题
        self.assertEqual(self.optimizer.problem.M, 1)  # 单目标优化

    def test_3d_distance_calculation(self):
        """测试三维距离计算"""
        model = DistanceDegradationModel()

        # 测试相同位置距离为0
        distance = model.calculate_3d_distance((100, 100, 100), (100, 100, 100))
        self.assertEqual(distance, 0)

        # 测试简单距离计算
        distance = model.calculate_3d_distance((0, 0, 0), (3, 4, 0))
        self.assertEqual(distance, 5)

        # 测试包含高度的距离计算
        distance = model.calculate_3d_distance((0, 0, 0), (0, 0, 10))
        self.assertEqual(distance, 10)

    def test_optimization_run(self):
        """测试优化运行"""
        # 运行优化
        result = self.optimizer.run_optimization()

        # 测试结果包含必要的键
        self.assertIn('best_ground_station_position', result)
        self.assertIn('best_fitness', result)
        self.assertIn('uav_degradations', result)
        self.assertIn('search_area', result)

        # 测试地面站位置在搜索区域内
        gs_pos = result['best_ground_station_position']
        search_area = result['search_area']
        self.assertGreaterEqual(gs_pos[0], search_area[0])
        self.assertLessEqual(gs_pos[0], search_area[2])
        self.assertGreaterEqual(gs_pos[1], search_area[1])
        self.assertLessEqual(gs_pos[1], search_area[3])

        # 测试地面站高度在允许范围内
        self.assertGreaterEqual(gs_pos[2], self.optimizer.gs_height_range[0])
        self.assertLessEqual(gs_pos[2], self.optimizer.gs_height_range[1])

        # 测试适应度值在0-1范围内
        self.assertGreaterEqual(result['best_fitness'], 0)
        self.assertLessEqual(result['best_fitness'], 1)

        # 测试每个无人机的劣化程度计算
        for uav_info in result['uav_degradations']:
            self.assertIn('position', uav_info)
            self.assertIn('distance', uav_info)
            self.assertIn('height', uav_info)
            self.assertIn('degradation', uav_info)
            self.assertGreaterEqual(uav_info['degradation'], 0)
            self.assertLessEqual(uav_info['degradation'], 1)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def setUp(self):
        self.model = DistanceDegradationModel()

    def test_zero_distance(self):
        """测试零距离情况"""
        # 零距离应该导致极高的路径损耗
        pl = self.model.path_loss(0)
        self.assertEqual(pl, float('inf'))

        # 零距离的劣化程度计算
        degradation = self.model.calculate_single_uav_degradation(0, 150)
        # 应该是一个很高的值，但由于归一化可能被限制在1
        self.assertGreater(degradation, 0.5)

    def test_extreme_heights(self):
        """测试极端高度"""
        # 测试负高度（不应该出现，但代码应该能处理）
        penalty_negative = self.model.height_penalty(-100)
        self.assertGreaterEqual(penalty_negative, 0)

        # 测试极高高度
        penalty_extreme = self.model.height_penalty(1000)
        self.assertLessEqual(penalty_extreme, 1)  # 应该被限制在1

    def test_single_uav_case(self):
        """测试单个无人机的情况"""
        uav_positions = [(500, 500, 150)]
        optimizer = DistanceOptimizer(
            uav_positions=uav_positions,
            population_size=5,
            max_generations=3
        )

        result = optimizer.run_optimization()

        # 应该能正常完成优化
        self.assertIn('best_ground_station_position', result)
        self.assertEqual(len(result['uav_degradations']), 1)

    def test_identical_uavs(self):
        """测试相同位置的多个无人机"""
        uav_positions = [
            (300, 300, 200),
            (300, 300, 200),
            (300, 300, 200)
        ]

        optimizer = DistanceOptimizer(
            uav_positions=uav_positions,
            population_size=5,
            max_generations=3
        )

        result = optimizer.run_optimization()

        # 所有无人机的劣化程度应该相同
        degradations = [uav['degradation'] for uav in result['uav_degradations']]
        self.assertEqual(len(set(round(d, 4) for d in degradations)), 1)


class TestPerformance(unittest.TestCase):
    """测试性能相关功能"""

    def test_large_population(self):
        """测试大种群性能"""
        # 创建多个无人机位置
        np.random.seed(42)
        uav_positions = []
        for i in range(20):
            x = np.random.uniform(0, 1000)
            y = np.random.uniform(0, 1000)
            h = np.random.uniform(50, 400)
            uav_positions.append((x, y, h))

        # 使用中等种群大小
        optimizer = DistanceOptimizer(
            uav_positions=uav_positions,
            population_size=20,
            max_generations=10
        )

        # 测试是否能正常完成优化
        result = optimizer.run_optimization()
        self.assertIsNotNone(result)
        self.assertEqual(len(result['uav_degradations']), 20)


def run_all_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestDistanceDegradationModel))
    suite.addTests(loader.loadTestsFromTestCase(TestDistanceOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    print("开始运行 UAV 通信优化系统单元测试...")
    print("=" * 60)

    success = run_all_tests()

    print("=" * 60)
    if success:
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败!")

    sys.exit(0 if success else 1)