"""
三维遮挡优化模块测试
测试ObstructionDegradationModel和ObstructionOptimizer3D的功能
"""

import unittest
import numpy as np
import sys
import os

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obstruction import (
    ObstructionDegradationModel,
    ObstructionEffectModel3D,
    ObstructionOptimizer3D
)


class TestObstructionDegradationModel(unittest.TestCase):
    """测试遮挡劣化模型"""

    def setUp(self):
        """测试前准备"""
        self.model = ObstructionDegradationModel()
        self.test_obstruction_levels = [0.0, 0.3, 0.6, 0.9, 1.0]

    def test_multipath_loss_calculation(self):
        """测试多径损耗计算"""
        # 测试遮挡程度为0时多径损耗为0
        mp_zero = self.model.multipath_loss(0.0)
        self.assertEqual(mp_zero, 0.0)

        # 测试遮挡程度增加时多径损耗增加
        mp_low = self.model.multipath_loss(0.3)
        mp_high = self.model.multipath_loss(0.8)
        self.assertGreater(mp_high, mp_low)

    def test_packet_loss_rate_range(self):
        """测试丢包率在合理范围内"""
        for level in self.test_obstruction_levels:
            plr = self.model.packet_loss_rate(level)
            self.assertGreaterEqual(plr, 0)
            self.assertLessEqual(plr, 1)

    def test_outage_probability_range(self):
        """测试中断概率在合理范围内"""
        for level in self.test_obstruction_levels:
            outage = self.model.outage_probability(level)
            self.assertGreaterEqual(outage, 0)
            self.assertLessEqual(outage, 1)

    def test_outage_probability_monotonicity(self):
        """测试中断概率随遮挡程度增加而增加"""
        outage_low = self.model.outage_probability(0.2)
        outage_high = self.model.outage_probability(0.8)
        self.assertGreater(outage_high, outage_low)

    def test_degradation_range(self):
        """测试劣化程度在0-1范围内"""
        for level in self.test_obstruction_levels:
            degradation = self.model.calculate_single_uav_degradation(level)
            self.assertGreaterEqual(degradation, 0)
            self.assertLessEqual(degradation, 1)

    def test_degradation_monotonicity(self):
        """测试劣化程度随遮挡程度增加而增加"""
        deg_low = self.model.calculate_single_uav_degradation(0.2)
        deg_high = self.model.calculate_single_uav_degradation(0.8)
        self.assertGreater(deg_high, deg_low)

    def test_weight_sum(self):
        """测试权重总和为1"""
        total_weight = sum(self.model.beta_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)

    def test_zero_obstruction(self):
        """测试无遮挡情况"""
        degradation = self.model.calculate_single_uav_degradation(0.0)
        # 无遮挡时应该有基础劣化程度
        self.assertGreaterEqual(degradation, 0)
        self.assertLess(degradation, 0.5)  # 应该相对较低


class TestObstructionEffectModel3D(unittest.TestCase):
    """测试三维遮挡效应模型"""

    def setUp(self):
        """测试前准备"""
        self.uav_positions = [(100, 200, 50), (300, 400, 80), (500, 600, 120)]
        self.model = ObstructionEffectModel3D(self.uav_positions)

    def test_obstruction_effect_calculation(self):
        """测试遮挡效应计算"""
        obstruction_pos = (200, 200, 60)
        obstruction_levels = self.model.calculate_obstruction_effect(obstruction_pos)

        # 检查返回列表长度正确
        self.assertEqual(len(obstruction_levels), len(self.uav_positions))

        # 检查所有遮挡程度在0-1范围内
        for level in obstruction_levels:
            self.assertGreaterEqual(level, 0)
            self.assertLessEqual(level, 1)

    def test_obstruction_effect_distance_dependency(self):
        """测试遮挡效应随距离衰减"""
        # 遮挡物靠近第一个无人机，远离第三个无人机
        obstruction_pos = (110, 210, 55)  # 靠近第一个无人机

        obstruction_levels = self.model.calculate_obstruction_effect(obstruction_pos)

        # 第一个无人机应该受到更强的遮挡
        self.assertGreater(obstruction_levels[0], obstruction_levels[2])

    def test_obstruction_strength_effect(self):
        """测试遮挡物强度的影响"""
        obstruction_pos = (200, 200, 60)

        levels_low = self.model.calculate_obstruction_effect(obstruction_pos, obstruction_strength=0.5)
        levels_high = self.model.calculate_obstruction_effect(obstruction_pos, obstruction_strength=1.0)

        # 高强度遮挡应该产生更高的遮挡程度
        for low, high in zip(levels_low, levels_high):
            self.assertGreaterEqual(high, low)

    def test_height_factor_calculation(self):
        """测试高度因子计算"""
        # 无人机高度低于遮挡物总高度时，高度因子应该为1
        height_factor = self.model._calculate_height_factor(50, 40, 20)  # 无人机50m, 遮挡物总高度60m
        self.assertEqual(height_factor, 1.0)

        # 无人机高度远高于遮挡物时，高度因子应该较小
        height_factor_low = self.model._calculate_height_factor(150, 40, 20)  # 无人机150m, 遮挡物总高度60m
        self.assertLess(height_factor_low, 0.5)

    def test_line_of_sight_calculation(self):
        """测试视线计算"""
        # 无人机高度低于遮挡物总高度，应该存在视线遮挡
        uav_pos = (100, 100, 50)
        obstruction_pos = (100, 100, 40)  # 基座高度40m
        obstruction_height = 20  # 遮挡物高度20m，总高度60m

        los_blocked = self.model.calculate_line_of_sight(uav_pos, obstruction_pos, obstruction_height)
        self.assertTrue(los_blocked)

        # 无人机高度高于遮挡物总高度，应该无视线遮挡
        uav_pos_high = (100, 100, 70)
        los_not_blocked = self.model.calculate_line_of_sight(uav_pos_high, obstruction_pos, obstruction_height)
        self.assertFalse(los_not_blocked)

    def test_far_obstruction(self):
        """测试远处遮挡物的影响"""
        # 遮挡物在很远的位置
        obstruction_pos = (2000, 2000, 100)
        obstruction_levels = self.model.calculate_obstruction_effect(obstruction_pos)

        # 远处的遮挡物应该几乎没有影响
        for level in obstruction_levels:
            self.assertAlmostEqual(level, 0.0, places=1)


class TestObstructionOptimizer3D(unittest.TestCase):
    """测试三维遮挡优化器"""

    def setUp(self):
        """测试前准备"""
        # 创建测试无人机位置（三维）
        self.uav_positions = [
            (100, 200, 50), (300, 400, 80), (500, 600, 120), (700, 800, 90)
        ]

        # 创建优化器（使用小种群和少代数以加快测试）
        self.optimizer = ObstructionOptimizer3D(
            uav_positions=self.uav_positions,
            population_size=20,
            max_generations=10,
            obstruction_strength=0.8,
            obstruction_height=50.0
        )

    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        self.assertEqual(len(self.optimizer.uav_positions), 4)
        self.assertEqual(self.optimizer.population_size, 20)
        self.assertEqual(self.optimizer.max_generations, 10)
        self.assertEqual(self.optimizer.obstruction_strength, 0.8)
        self.assertEqual(self.optimizer.obstruction_height, 50.0)
        self.assertIsNotNone(self.optimizer.degradation_model)
        self.assertIsNotNone(self.optimizer.obstruction_model)
        self.assertIsNotNone(self.optimizer.problem)
        self.assertIsNotNone(self.optimizer.algorithm)

    def test_search_area_calculation(self):
        """测试三维搜索区域自动计算"""
        search_area = self.optimizer.problem.search_area
        self.assertEqual(len(search_area), 6)  # x_min, y_min, z_min, x_max, y_max, z_max

        # 检查搜索区域包含所有无人机
        uav_array = np.array(self.uav_positions)
        x_min, y_min, z_min = np.min(uav_array, axis=0)
        x_max, y_max, z_max = np.max(uav_array, axis=0)

        self.assertLess(search_area[0], x_min)  # x_min 应该小于无人机最小x
        self.assertLess(search_area[1], y_min)  # y_min 应该小于无人机最小y
        self.assertLess(search_area[2], z_min)  # z_min 应该小于无人机最小z
        self.assertGreater(search_area[3], x_max)  # x_max 应该大于无人机最大x
        self.assertGreater(search_area[4], y_max)  # y_max 应该大于无人机最大y
        self.assertGreater(search_area[5], z_max)  # z_max 应该大于无人机最大z

    def test_optimization_run(self):
        """测试优化运行"""
        result = self.optimizer.run_optimization()

        # 检查结果包含必要的键
        required_keys = [
            'best_obstruction_position',
            'best_fitness',
            'obstruction_strength',
            'obstruction_height',
            'uav_degradations',
            'average_obstruction_level',
            'max_obstruction_level',
            'los_blocked_count',
            'search_area',
            'optimization_parameters'
        ]

        for key in required_keys:
            self.assertIn(key, result)

        # 检查最优位置在搜索区域内
        best_pos = result['best_obstruction_position']
        search_area = result['search_area']
        self.assertGreaterEqual(best_pos[0], search_area[0])
        self.assertLessEqual(best_pos[0], search_area[3])
        self.assertGreaterEqual(best_pos[1], search_area[1])
        self.assertLessEqual(best_pos[1], search_area[4])
        self.assertGreaterEqual(best_pos[2], search_area[2])
        self.assertLessEqual(best_pos[2], search_area[5])

        # 检查适应度值在合理范围内
        fitness = result['best_fitness']
        self.assertGreaterEqual(fitness, 0)
        self.assertLessEqual(fitness, 1)

        # 检查遮挡程度统计
        self.assertGreaterEqual(result['average_obstruction_level'], 0)
        self.assertLessEqual(result['average_obstruction_level'], 1)
        self.assertGreaterEqual(result['max_obstruction_level'], 0)
        self.assertLessEqual(result['max_obstruction_level'], 1)

        # 检查视线遮挡统计
        self.assertGreaterEqual(result['los_blocked_count'], 0)
        self.assertLessEqual(result['los_blocked_count'], len(self.uav_positions))

        # 检查每个无人机的劣化信息
        self.assertEqual(len(result['uav_degradations']), len(self.uav_positions))
        for uav_info in result['uav_degradations']:
            required_uav_keys = ['uav_id', 'position', 'distance_to_obstruction',
                                 'obstruction_level', 'degradation', 'line_of_sight_blocked']
            for key in required_uav_keys:
                self.assertIn(key, uav_info)

            self.assertGreaterEqual(uav_info['distance_to_obstruction'], 0)
            self.assertGreaterEqual(uav_info['obstruction_level'], 0)
            self.assertLessEqual(uav_info['obstruction_level'], 1)
            self.assertGreaterEqual(uav_info['degradation'], 0)
            self.assertLessEqual(uav_info['degradation'], 1)
            self.assertIn(uav_info['line_of_sight_blocked'], [True, False])


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_single_uav_optimization(self):
        """测试单无人机优化"""
        uav_positions = [(500, 500, 100)]
        optimizer = ObstructionOptimizer3D(
            uav_positions=uav_positions,
            population_size=10,
            max_generations=5,
            obstruction_strength=0.5,
            obstruction_height=30.0
        )
        result = optimizer.run_optimization()

        # 单无人机情况下，最优位置应该接近无人机位置以获得最大遮挡
        best_pos = result['best_obstruction_position']
        uav_pos = uav_positions[0]
        distance = np.linalg.norm(np.array(best_pos) - np.array(uav_pos))
        self.assertLess(distance, 100)  # 应该比较接近

    def test_very_weak_obstruction(self):
        """测试很弱的遮挡物"""
        uav_positions = [(100, 100, 50), (300, 300, 80)]
        optimizer = ObstructionOptimizer3D(
            uav_positions=uav_positions,
            population_size=15,
            max_generations=5,
            obstruction_strength=0.1,  # 很弱的遮挡
            obstruction_height=40.0
        )
        result = optimizer.run_optimization()

        # 弱遮挡应该产生较低的劣化程度
        self.assertLess(result['best_fitness'], 0.5)

    def test_very_strong_obstruction(self):
        """测试很强的遮挡物"""
        uav_positions = [(100, 100, 60), (300, 300, 90)]
        optimizer = ObstructionOptimizer3D(
            uav_positions=uav_positions,
            population_size=15,
            max_generations=5,
            obstruction_strength=1.0,  # 很强的遮挡
            obstruction_height=60.0
        )
        result = optimizer.run_optimization()

        # 强遮挡应该产生较高的劣化程度
        self.assertGreater(result['best_fitness'], 0.3)

    def test_high_obstruction(self):
        """测试高遮挡物"""
        uav_positions = [(100, 100, 50), (300, 300, 80)]
        optimizer = ObstructionOptimizer3D(
            uav_positions=uav_positions,
            population_size=15,
            max_generations=5,
            obstruction_strength=0.8,
            obstruction_height=100.0  # 很高的遮挡物
        )
        result = optimizer.run_optimization()

        # 高遮挡物应该产生更多的视线遮挡
        self.assertGreaterEqual(result['los_blocked_count'], 1)


class TestIntegration(unittest.TestCase):
    """测试集成功能"""

    def test_summary_generation(self):
        """测试结果摘要生成"""
        uav_positions = [(100, 100, 50), (200, 200, 70)]
        optimizer = ObstructionOptimizer3D(
            uav_positions=uav_positions,
            population_size=10,
            max_generations=5,
            obstruction_height=40.0
        )
        result = optimizer.run_optimization()

        summary = optimizer.get_optimization_summary()

        # 检查摘要包含关键信息
        self.assertIn("最优遮挡物位置", summary)
        self.assertIn("平均通信劣化程度", summary)
        self.assertIn("无人机", summary)
        self.assertIn("视线遮挡", summary)

    def test_plotting_function(self):
        """测试绘图功能（不实际显示）"""
        uav_positions = [(100, 100, 50), (200, 200, 70), (300, 300, 90)]
        optimizer = ObstructionOptimizer3D(
            uav_positions=uav_positions,
            population_size=10,
            max_generations=5,
            obstruction_height=45.0
        )
        result = optimizer.run_optimization()

        # 只是检查绘图函数是否能正常调用而不出错
        try:
            # 我们不在测试中实际显示图表
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            optimizer.plot_optimization_results()
            plot_success = True
        except Exception as e:
            plot_success = False
            print(f"绘图测试中出现错误: {e}")

        self.assertTrue(plot_success)


def run_performance_test():
    """运行性能测试（非单元测试）"""
    print("运行三维遮挡优化性能测试...")

    # 创建大量无人机测试性能
    np.random.seed(42)
    num_uavs = 20
    uav_positions = [(np.random.uniform(0, 1000), np.random.uniform(0, 1000), np.random.uniform(50, 150))
                     for _ in range(num_uavs)]

    import time
    start_time = time.time()

    optimizer = ObstructionOptimizer3D(
        uav_positions=uav_positions,
        population_size=30,
        max_generations=15,
        obstruction_strength=0.7,
        obstruction_height=60.0
    )

    result = optimizer.run_optimization()
    end_time = time.time()

    print(f"性能测试完成:")
    print(f"  无人机数量: {num_uavs}")
    print(f"  运行时间: {end_time - start_time:.2f}秒")
    print(f"  最优劣化程度: {result['best_fitness']:.4f}")
    print(f"  平均遮挡程度: {result['average_obstruction_level']:.4f}")
    print(f"  视线遮挡数量: {result['los_blocked_count']}/{num_uavs}")

    return result


if __name__ == '__main__':
    # 运行单元测试
    print("运行三维遮挡优化模块单元测试...")
    unittest.main(verbosity=2)

    # 可选：运行性能测试
    # print("\n" + "="*50)
    # run_performance_test()

    # 可选：运行快速测试
    # from obstruction import quick_test
    # quick_test()