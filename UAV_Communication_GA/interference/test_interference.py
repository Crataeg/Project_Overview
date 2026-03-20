"""
无人机通信干扰优化系统 - 单元测试
测试干扰模型、优化算法的正确性，包括高度因素
"""

import unittest
import numpy as np
import sys
import os

# 添加当前目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from interference import InterferenceDegradationModel, InterferenceOptimizer


class TestInterferenceDegradationModel(unittest.TestCase):
    """测试干扰劣化程度计算模型"""

    def setUp(self):
        """测试前初始化"""
        self.model = InterferenceDegradationModel()

    def test_3d_distance_calculation(self):
        """测试三维距离计算"""
        # 测试相同位置距离为0
        distance = self.model.calculate_3d_distance((100, 100, 100), (100, 100, 100))
        self.assertEqual(distance, 0)

        # 测试简单距离计算
        distance = self.model.calculate_3d_distance((0, 0, 0), (3, 4, 0))
        self.assertEqual(distance, 5)

        # 测试包含高度的距离计算
        distance = self.model.calculate_3d_distance((0, 0, 0), (0, 0, 10))
        self.assertEqual(distance, 10)

        # 测试复杂三维距离
        distance = self.model.calculate_3d_distance((0, 0, 0), (3, 4, 12))
        self.assertEqual(distance, 13)  # 3-4-5三角形扩展为3-4-12-13

    def test_height_penalty(self):
        """测试高度惩罚函数"""
        # 测试最优高度附近的惩罚
        penalty_optimal = self.model.height_penalty(self.model.interference_params['h_optimal'])
        self.assertLess(penalty_optimal, 0.1)

        # 测试高度偏离时的惩罚增加
        penalty_low = self.model.height_penalty(50)
        penalty_high = self.model.height_penalty(400)
        self.assertGreater(penalty_high, penalty_low)

        # 测试惩罚值在0-1范围内
        self.assertGreaterEqual(penalty_optimal, 0)
        self.assertLessEqual(penalty_optimal, 1)

        # 测试边界情况
        penalty_max = self.model.height_penalty(self.model.interference_params['h_max'])
        self.assertLessEqual(penalty_max, 1.0)

    def test_spectrum_overlap_factor(self):
        """测试频谱重叠因子计算"""
        # 测试相同频率时的重叠因子
        f_uav = self.model.interference_params['f_uav']
        overlap_same = self.model.spectrum_overlap_factor(f_uav, 20)
        self.assertAlmostEqual(overlap_same, 1.0, delta=0.01)

        # 测试频率偏离时的重叠因子减小
        overlap_far = self.model.spectrum_overlap_factor(f_uav + 1000, 20)
        self.assertLess(overlap_far, overlap_same)

        # 测试重叠因子在0-1范围内
        self.assertGreaterEqual(overlap_same, 0)
        self.assertLessEqual(overlap_same, 1)

    def test_interference_loss(self):
        """测试干扰损耗计算"""
        # 测试距离增加时干扰损耗减小
        loss_close = self.model.interference_loss(100, 2400, 20, 1.0, 150)
        loss_far = self.model.interference_loss(1000, 2400, 20, 1.0, 150)
        self.assertLess(loss_far, loss_close)

        # 测试高度偏离最优值时干扰损耗增加
        loss_optimal = self.model.interference_loss(500, 2400, 20, 1.0, 150)
        loss_high = self.model.interference_loss(500, 2400, 20, 1.0, 400)
        self.assertGreater(loss_high, loss_optimal)

        # 测试功率增加时干扰损耗增加
        loss_low_power = self.model.interference_loss(500, 2400, 20, 0.5, 150)
        loss_high_power = self.model.interference_loss(500, 2400, 20, 2.0, 150)
        self.assertGreater(loss_high_power, loss_low_power)

    def test_total_interference_power(self):
        """测试总干扰功率计算"""
        # 测试总干扰功率在0-1范围内
        total_power = self.model.total_interference_power(500, 2400, 20, 1.0, 150)
        self.assertGreaterEqual(total_power, 0)
        self.assertLessEqual(total_power, 1)

        # 测试距离增加时总干扰功率减小
        power_close = self.model.total_interference_power(100, 2400, 20, 1.0, 150)
        power_far = self.model.total_interference_power(1000, 2400, 20, 1.0, 150)
        self.assertLess(power_far, power_close)

    def test_packet_loss_rate(self):
        """测试干扰丢包率计算"""
        # 测试丢包率在0-1范围内
        plr = self.model.packet_loss_rate(0.5, 150)
        self.assertGreaterEqual(plr, 0)
        self.assertLessEqual(plr, 1)

        # 测试干扰增加时丢包率增加
        plr_low = self.model.packet_loss_rate(0.2, 150)
        plr_high = self.model.packet_loss_rate(0.8, 150)
        self.assertGreater(plr_high, plr_low)

    def test_outage_probability(self):
        """测试中断概率计算"""
        # 测试中断概率在0-1范围内
        p_out = self.model.outage_probability(0.5, 150)
        self.assertGreaterEqual(p_out, 0)
        self.assertLessEqual(p_out, 1)

        # 测试干扰增加时中断概率增加
        p_out_low = self.model.outage_probability(0.2, 150)
        p_out_high = self.model.outage_probability(0.8, 150)
        self.assertGreater(p_out_high, p_out_low)

    def test_degradation_calculation(self):
        """测试劣化程度计算"""
        # 测试劣化程度在0-1范围内
        degradation = self.model.calculate_single_uav_degradation(500, 2400, 20, 1.0, 200)
        self.assertGreaterEqual(degradation, 0)
        self.assertLessEqual(degradation, 1)

        # 测试距离增加时劣化程度减小
        deg_close = self.model.calculate_single_uav_degradation(100, 2400, 20, 1.0, 150)
        deg_far = self.model.calculate_single_uav_degradation(1500, 2400, 20, 1.0, 150)
        self.assertLess(deg_far, deg_close)

        # 测试高度偏离最优值时劣化程度增加
        deg_optimal = self.model.calculate_single_uav_degradation(500, 2400, 20, 1.0, 150)
        deg_high = self.model.calculate_single_uav_degradation(500, 2400, 20, 1.0, 400)
        self.assertGreater(deg_high, deg_optimal)

        # 测试频率匹配时劣化程度增加
        deg_match = self.model.calculate_single_uav_degradation(500, 2400, 20, 1.0, 150)
        deg_mismatch = self.model.calculate_single_uav_degradation(500, 5000, 20, 1.0, 150)
        self.assertGreater(deg_match, deg_mismatch)

    def test_interference_band_info(self):
        """测试干扰频段信息获取"""
        # 测试在频段内的匹配
        band_name, match_quality = self.model.get_interference_band_info(2450)
        self.assertEqual(band_name, "WiFi")
        self.assertAlmostEqual(match_quality, 1.0, delta=0.01)

        # 测试频段外的匹配
        band_name, match_quality = self.model.get_interference_band_info(3000)
        self.assertLess(match_quality, 1.0)

        # 测试匹配度在0-1范围内
        self.assertGreaterEqual(match_quality, 0)
        self.assertLessEqual(match_quality, 1)


class TestInterferenceOptimizer(unittest.TestCase):
    """测试干扰优化器"""

    def setUp(self):
        """测试前初始化"""
        # 创建测试用的无人机位置（包含高度）
        self.uav_positions = [
            (100, 100, 150),
            (200, 200, 200),
            (300, 100, 100),
            (100, 300, 250),
            (250, 250, 180)
        ]

        # 使用较小的种群和代数进行快速测试
        self.optimizer = InterferenceOptimizer(
            uav_positions=self.uav_positions,
            population_size=10,
            max_generations=5,
            jammer_height_range=(0, 30)
        )

    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        self.assertEqual(len(self.optimizer.uav_positions), 5)
        self.assertEqual(self.optimizer.population_size, 10)
        self.assertEqual(self.optimizer.max_generations, 5)
        self.assertEqual(self.optimizer.jammer_height_range, (0, 30))

        # 测试问题定义
        self.assertEqual(self.optimizer.problem.Dim, 6)  # 六维优化问题
        self.assertEqual(self.optimizer.problem.M, 1)  # 单目标优化

        # 测试参数范围
        self.assertEqual(self.optimizer.problem.frequency_range, (100, 6000))
        self.assertEqual(self.optimizer.problem.bandwidth_range, (1, 100))
        self.assertEqual(self.optimizer.problem.power_range, (0.1, 10.0))

    def test_optimization_run(self):
        """测试优化运行"""
        # 运行优化
        result = self.optimizer.run_optimization()

        # 测试结果包含必要的键
        required_keys = [
            'best_interference_position', 'best_frequency', 'best_bandwidth',
            'best_power', 'interference_band', 'frequency_match_quality',
            'total_interference_power', 'best_fitness', 'uav_degradations',
            'search_area', 'jammer_height_range', 'parameter_ranges',
            'optimization_parameters'
        ]
        for key in required_keys:
            self.assertIn(key, result)

        # 测试干扰源位置在搜索区域内
        jammer_pos = result['best_interference_position']
        search_area = result['search_area']
        self.assertGreaterEqual(jammer_pos[0], search_area[0])
        self.assertLessEqual(jammer_pos[0], search_area[2])
        self.assertGreaterEqual(jammer_pos[1], search_area[1])
        self.assertLessEqual(jammer_pos[1], search_area[3])

        # 测试干扰源高度在允许范围内
        self.assertGreaterEqual(jammer_pos[2], self.optimizer.jammer_height_range[0])
        self.assertLessEqual(jammer_pos[2], self.optimizer.jammer_height_range[1])

        # 测试频率在允许范围内
        self.assertGreaterEqual(result['best_frequency'], 100)
        self.assertLessEqual(result['best_frequency'], 6000)

        # 测试带宽在允许范围内
        self.assertGreaterEqual(result['best_bandwidth'], 1)
        self.assertLessEqual(result['best_bandwidth'], 100)

        # 测试功率在允许范围内
        self.assertGreaterEqual(result['best_power'], 0.1)
        self.assertLessEqual(result['best_power'], 10.0)

        # 测试适应度值在0-1范围内
        self.assertGreaterEqual(result['best_fitness'], 0)
        self.assertLessEqual(result['best_fitness'], 1)

        # 测试每个无人机的劣化程度计算
        for uav_info in result['uav_degradations']:
            self.assertIn('uav_id', uav_info)
            self.assertIn('position', uav_info)
            self.assertIn('distance_to_interference', uav_info)
            self.assertIn('height', uav_info)
            self.assertIn('degradation', uav_info)
            self.assertEqual(len(uav_info['position']), 3)  # 三维位置
            self.assertGreaterEqual(uav_info['degradation'], 0)
            self.assertLessEqual(uav_info['degradation'], 1)

    def test_optimization_summary(self):
        """测试优化结果摘要"""
        # 运行优化
        result = self.optimizer.run_optimization()

        # 获取摘要
        summary = self.optimizer.get_optimization_summary()

        # 测试摘要包含关键信息
        self.assertIn("干扰优化结果摘要", summary)
        self.assertIn("最优干扰源位置", summary)
        self.assertIn("最优干扰频率", summary)
        self.assertIn("平均通信劣化程度", summary)
        self.assertIn("各无人机详情", summary)

        # 测试摘要包含所有无人机信息
        for uav_info in result['uav_degradations']:
            self.assertIn(f"无人机 {uav_info['uav_id']}", summary)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def setUp(self):
        self.model = InterferenceDegradationModel()

    def test_zero_distance(self):
        """测试零距离情况"""
        # 零距离应该导致极高的干扰
        total_power = self.model.total_interference_power(0, 2400, 20, 1.0, 150)
        degradation = self.model.calculate_single_uav_degradation(0, 2400, 20, 1.0, 150)

        # 应该是一个很高的值
        self.assertGreater(total_power, 0.5)
        self.assertGreater(degradation, 0.5)

    def test_extreme_heights(self):
        """测试极端高度"""
        # 测试负高度（不应该出现，但代码应该能处理）
        penalty_negative = self.model.height_penalty(-100)
        self.assertGreaterEqual(penalty_negative, 0)

        # 测试极高高度
        penalty_extreme = self.model.height_penalty(1000)
        self.assertLessEqual(penalty_extreme, 1)  # 应该被限制在1

        # 测试极端高度的干扰计算
        degradation = self.model.calculate_single_uav_degradation(500, 2400, 20, 1.0, 1000)
        self.assertLessEqual(degradation, 1.0)

    def test_extreme_frequencies(self):
        """测试极端频率"""
        # 测试极低频率
        overlap_low = self.model.spectrum_overlap_factor(10, 20)
        self.assertGreaterEqual(overlap_low, 0)

        # 测试极高频率
        overlap_high = self.model.spectrum_overlap_factor(10000, 20)
        self.assertGreaterEqual(overlap_high, 0)

        # 测试极端频率的干扰计算
        degradation = self.model.calculate_single_uav_degradation(500, 10000, 20, 1.0, 150)
        self.assertLessEqual(degradation, 1.0)

    def test_single_uav_case(self):
        """测试单个无人机的情况"""
        uav_positions = [(500, 500, 150)]
        optimizer = InterferenceOptimizer(
            uav_positions=uav_positions,
            population_size=5,
            max_generations=3,
            jammer_height_range=(0, 30)
        )

        result = optimizer.run_optimization()

        # 应该能正常完成优化
        self.assertIn('best_interference_position', result)
        self.assertEqual(len(result['uav_degradations']), 1)

    def test_identical_uavs(self):
        """测试相同位置的多个无人机"""
        uav_positions = [
            (300, 300, 200),
            (300, 300, 200),
            (300, 300, 200)
        ]

        optimizer = InterferenceOptimizer(
            uav_positions=uav_positions,
            population_size=5,
            max_generations=3,
            jammer_height_range=(0, 30)
        )

        result = optimizer.run_optimization()

        # 所有无人机的劣化程度应该非常接近
        degradations = [uav['degradation'] for uav in result['uav_degradations']]
        max_diff = max(degradations) - min(degradations)
        self.assertLess(max_diff, 0.01)  # 允许微小差异


class TestPerformance(unittest.TestCase):
    """测试性能相关功能"""

    def test_large_population(self):
        """测试大种群性能"""
        # 创建多个无人机位置
        np.random.seed(42)
        uav_positions = []
        for i in range(15):
            x = np.random.uniform(0, 1000)
            y = np.random.uniform(0, 1000)
            h = np.random.uniform(50, 400)
            uav_positions.append((x, y, h))

        # 使用中等种群大小
        optimizer = InterferenceOptimizer(
            uav_positions=uav_positions,
            population_size=20,
            max_generations=10,
            jammer_height_range=(0, 30)
        )

        # 测试是否能正常完成优化
        result = optimizer.run_optimization()
        self.assertIsNotNone(result)
        self.assertEqual(len(result['uav_degradations']), 15)

    def test_plotting_functions(self):
        """测试绘图功能（不显示，只检查是否正常运行）"""
        # 创建测试数据
        uav_positions = [
            (100, 100, 150),
            (200, 200, 200),
            (300, 100, 100)
        ]

        optimizer = InterferenceOptimizer(
            uav_positions=uav_positions,
            population_size=5,
            max_generations=3,
            jammer_height_range=(0, 30)
        )

        # 运行优化
        result = optimizer.run_optimization()

        # 测试绘图函数不抛出异常
        try:
            optimizer.plot_optimization_results()
            optimizer.plot_frequency_analysis()
        except Exception as e:
            self.fail(f"绘图函数抛出异常: {e}")


class TestIntegration(unittest.TestCase):
    """测试集成功能"""

    def test_interference_bands_coverage(self):
        """测试干扰频段覆盖"""
        model = InterferenceDegradationModel()

        # 测试所有预定义频段
        test_frequencies = {
            2450: "WiFi",
            2600: "4G",
            3500: "5G",
            450: "Public_Safety",
            2100: "Military",
            950: "Industrial"
        }

        for freq, expected_band in test_frequencies.items():
            band_name, match_quality = model.get_interference_band_info(freq)
            self.assertEqual(band_name, expected_band)
            self.assertGreater(match_quality, 0.9)  # 在频段内应该有高匹配度

    def test_parameter_consistency(self):
        """测试参数一致性"""
        model = InterferenceDegradationModel()

        # 测试高度参数与distance.py一致
        self.assertEqual(model.interference_params['h_max'], 500)
        self.assertEqual(model.interference_params['h_optimal'], 150)
        self.assertEqual(model.interference_params['h_penalty_threshold'], 300)

        # 测试权重系数和为1（允许微小误差）
        total_weight = sum(model.gamma_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, delta=0.001)


def run_all_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestInterferenceDegradationModel))
    suite.addTests(loader.loadTestsFromTestCase(TestInterferenceOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    print("开始运行 UAV 干扰优化系统单元测试...")
    print("=" * 60)

    success = run_all_tests()

    print("=" * 60)
    if success:
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败!")

    sys.exit(0 if success else 1)