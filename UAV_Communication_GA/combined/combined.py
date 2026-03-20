"""
综合联调系统 - 组合距离、遮挡、干扰、速度四个中间件进行全局优化（三维版本）
总通信劣化程度 = W₁ × D₁(d) + W₂ × D₂(o) + W₃ × D₃(i) + W₄ × D₄(v)
"""

import numpy as np
import geatpy as ea
import sys
import os

# 添加模块路径，确保可以导入其他目录的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入其他中间件的模型
try:
    from distance.distance import DistanceOptimizer
    from obstruction.obstruction import ObstructionOptimizer3D
    from interference.interference import InterferenceOptimizer
except ImportError as e:
    print(f"警告: 无法导入某些中间件: {e}")
    print("将使用简化的内部模型")


class CombinedDegradationModel:
    """综合劣化模型 - 组合四个因素的劣化程度（三维版本）"""

    def __init__(self, uav_positions):
        self.uav_positions = np.array(uav_positions)
        self.num_uavs = len(uav_positions)

        # 综合权重配置 - 可根据实际情况调整
        self.combined_weights = {
            'distance': 0.3,  # W₁ - 距离权重
            'obstruction': 0.2,  # W₂ - 遮挡权重
            'interference': 0.3,  # W₃ - 干扰权重
            'velocity': 0.2  # W₄ - 速度权重
        }

        # 决策变量范围定义（三维版本）
        self.var_ranges = {
            # 地面站位置 (2维) - 距离因素
            'gs_x': [0, 1000],
            'gs_y': [0, 1000],

            # 遮挡物位置 (3维) - 遮挡因素
            'obs_x': [0, 1000],
            'obs_y': [0, 1000],
            'obs_z': [0, 100],  # 遮挡物高度

            # 干扰源参数 (6维) - 干扰因素
            'int_x': [0, 1000],
            'int_y': [0, 1000],
            'int_z': [0, 100],  # 干扰源高度
            'int_freq': [100e6, 6000e6],  # 100MHz - 6GHz
            'int_bandwidth': [1e6, 100e6],  # 1MHz - 100MHz
            'int_power': [0.1, 10],  # 0.1W - 10W

            # 无人机速度 (每个无人机1维) - 速度因素
            'uav_velocities': [0, 50]  # 0-50 m/s
        }

    def set_combined_weights(self, weights):
        """设置综合权重"""
        self.combined_weights.update(weights)
        # 归一化权重，确保总和为1
        total = sum(self.combined_weights.values())
        for key in self.combined_weights:
            self.combined_weights[key] /= total
        print(f"更新后的综合权重: {self.combined_weights}")

    def calculate_3d_distance(self, pos1, pos2):
        """计算三维距离"""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 +
                       (pos1[1] - pos2[1]) ** 2 +
                       (pos1[2] - pos2[2]) ** 2)

    def calculate_distance_degradation(self, gs_position):
        """
        计算距离劣化程度 D₁(d) - 三维版本
        基于三维距离的通信劣化模型
        """
        total_deg = 0
        gs_pos_3d = [gs_position[0], gs_position[1], 0]  # 地面站高度设为0

        for uav_pos in self.uav_positions:
            # 计算地面站与无人机之间的三维距离
            distance_3d = self.calculate_3d_distance(gs_pos_3d, uav_pos)

            # 距离劣化模型 - 基于三维距离路径损耗
            if distance_3d <= 0:
                deg = 0.0
            else:
                # 简化的三维距离劣化模型
                reference_distance = 100  # 参考距离
                path_loss_exponent = 2.5  # 路径损耗指数

                if distance_3d > reference_distance:
                    path_loss = 20 * np.log10(4 * np.pi * reference_distance / 0.125) + \
                                10 * path_loss_exponent * np.log10(distance_3d / reference_distance)
                else:
                    path_loss = 20 * np.log10(4 * np.pi * distance_3d / 0.125)

                # 路径损耗导致的劣化
                path_loss_deg = min(path_loss / 120, 1.0)

                # 距离相关的丢包率
                packet_loss = 0.01 + 0.3 * (1 - np.exp(-distance_3d / 500))

                # 高度差影响
                height_diff = abs(uav_pos[2] - gs_pos_3d[2])
                height_effect = min(height_diff / 200, 0.3)  # 最大30%的影响

                # 综合距离劣化
                deg = 0.5 * path_loss_deg + 0.3 * packet_loss + 0.2 * height_effect

            total_deg += deg

        return min(total_deg / self.num_uavs, 1.0)

    def calculate_obstruction_degradation(self, obs_position):
        """
        计算遮挡劣化程度 D₂(o) - 三维版本
        基于三维遮挡因素的通信劣化模型
        """
        total_deg = 0
        obs_pos_3d = obs_position  # [x, y, z]

        for uav_pos in self.uav_positions:
            # 计算遮挡物到无人机的三维距离
            obs_distance_3d = self.calculate_3d_distance(obs_pos_3d, uav_pos)

            # 计算视线遮挡
            # 简化的视线判断：如果无人机高度低于遮挡物高度，则存在遮挡
            los_blocked = uav_pos[2] <= obs_pos_3d[2]

            # 遮挡劣化模型
            if obs_distance_3d <= 10:  # 非常接近
                deg = 1.0  # 完全遮挡
            else:
                # 遮挡效应随距离衰减
                obstruction_effect = np.exp(-obs_distance_3d / 150)

                # 视线遮挡增强效应
                los_enhancement = 1.5 if los_blocked else 1.0

                # 多径效应导致的劣化
                multipath_deg = 0.6 * obstruction_effect * los_enhancement

                # 信号阴影效应
                shadowing_deg = 0.3 * (1 - np.exp(-obs_distance_3d / 300))

                # 高度相关效应
                height_effect = 0.1 * (1 - abs(uav_pos[2] - obs_pos_3d[2]) / 100)

                deg = multipath_deg + shadowing_deg + height_effect

            total_deg += min(deg, 1.0)

        return min(total_deg / self.num_uavs, 1.0)

    def calculate_interference_degradation(self, int_params):
        """
        计算干扰劣化程度 D₃(i) - 三维版本
        基于三维干扰因素的通信劣化模型
        """
        total_deg = 0
        int_position_3d = int_params[:3]  # [x, y, z]
        frequency = int_params[3]
        bandwidth = int_params[4]
        power = int_params[5]

        for uav_pos in self.uav_positions:
            # 计算干扰源到无人机的三维距离
            int_distance_3d = self.calculate_3d_distance(int_position_3d, uav_pos)

            # 干扰劣化模型
            if int_distance_3d <= 10:
                deg = 1.0  # 完全干扰
            else:
                # 距离衰减（三维）
                distance_effect = 1 / (1 + int_distance_3d / 80)

                # 频谱重叠因子 (假设无人机工作在2.4GHz)
                uav_frequency = 2.4e9
                freq_overlap = 1 / (1 + abs(frequency - uav_frequency) / 100e6)

                # 功率影响
                power_effect = min(power / 8, 1.0)

                # 带宽影响
                bandwidth_effect = min(bandwidth / 80e6, 1.0)

                # 高度相关效应
                height_diff = abs(uav_pos[2] - int_position_3d[2])
                height_effect = 1 / (1 + height_diff / 50)

                # 综合干扰劣化
                deg = (0.3 * distance_effect + 0.25 * freq_overlap +
                       0.2 * power_effect + 0.15 * bandwidth_effect +
                       0.1 * height_effect)

            total_deg += min(deg, 1.0)

        return min(total_deg / self.num_uavs, 1.0)

    def calculate_velocity_degradation(self, uav_velocities):
        """
        计算速度劣化程度 D₄(v)
        基于速度因素的通信劣化模型（与高度无关）
        """
        total_deg = 0
        for velocity in uav_velocities:
            # 速度劣化模型
            if velocity <= 0:
                deg = 0.0
            else:
                # 多普勒频移效应
                doppler_shift = (velocity / 3e8) * 2.4e9  # 粗略估计
                doppler_deg = min(doppler_shift / 1000, 1.0)  # 归一化

                # 切换失败率随速度增加
                handover_failure = 0.1 + 0.4 * (1 - np.exp(-velocity / 20))

                # 信道变化率
                channel_variation = min(velocity / 40, 1.0)

                # 综合速度劣化
                deg = 0.3 * doppler_deg + 0.4 * handover_failure + 0.3 * channel_variation

            total_deg += deg

        return min(total_deg / self.num_uavs, 1.0)

    def calculate_combined_degradation(self, decision_vars):
        """
        计算综合通信劣化程度 - 三维版本
        总劣化程度 = W₁ × D₁(d) + W₂ × D₂(o) + W₃ × D₃(i) + W₄ × D₄(v)

        参数:
            decision_vars: 决策变量数组 [gs_x, gs_y, obs_x, obs_y, obs_z, int_x, int_y, int_z, int_freq, int_bandwidth, int_power, v1, v2, ..., vn]

        返回:
            total_degradation: 总劣化程度
            components: 各分量劣化程度
        """
        # 解析决策变量 - 三维版本
        gs_position = decision_vars[0:2]  # 地面站位置 - 距离因素
        obs_position = decision_vars[2:5]  # 遮挡物位置 (三维) - 遮挡因素
        int_params = decision_vars[5:11]  # 干扰源参数 (三维) - 干扰因素
        uav_velocities = decision_vars[11:11 + self.num_uavs]  # 各无人机速度 - 速度因素

        # 计算各分量劣化程度
        distance_degradation = self.calculate_distance_degradation(gs_position)
        obstruction_degradation = self.calculate_obstruction_degradation(obs_position)
        interference_degradation = self.calculate_interference_degradation(int_params)
        velocity_degradation = self.calculate_velocity_degradation(uav_velocities)

        # 加权综合 - 核心公式
        total_degradation = (
                self.combined_weights['distance'] * distance_degradation +
                self.combined_weights['obstruction'] * obstruction_degradation +
                self.combined_weights['interference'] * interference_degradation +
                self.combined_weights['velocity'] * velocity_degradation
        )

        components = {
            'distance': distance_degradation,
            'obstruction': obstruction_degradation,
            'interference': interference_degradation,
            'velocity': velocity_degradation,
            'contributions': [
                self.combined_weights['distance'] * distance_degradation,
                self.combined_weights['obstruction'] * obstruction_degradation,
                self.combined_weights['interference'] * interference_degradation,
                self.combined_weights['velocity'] * velocity_degradation
            ]
        }

        return min(total_degradation, 1.0), components


class CombinedOptimizer:
    """综合优化器 - 使用遗传算法寻找最优组合（三维版本）"""

    def __init__(self, uav_positions, population_size=100, max_generations=200):
        self.uav_positions = np.array(uav_positions)
        self.num_uavs = len(uav_positions)
        self.population_size = population_size
        self.max_generations = max_generations

        # 初始化综合劣化模型
        self.degradation_model = CombinedDegradationModel(uav_positions)

        # 决策变量总维度: 2(gs) + 3(obs) + 6(int) + n(uav_vel) = 11 + n
        self.total_dim = 11 + self.num_uavs

        # 决策变量边界
        self.ranges = self.degradation_model.var_ranges
        self.lb, self.ub = self._get_variable_bounds()

        # 优化历史记录
        self.optimization_history = {
            'best_fitness': [],
            'mean_fitness': [],
            'best_solutions': []
        }

    def _get_variable_bounds(self):
        """获取决策变量边界 - 三维版本"""
        lb = []
        ub = []

        # 地面站位置 (距离因素) - 2维
        lb.extend([self.ranges['gs_x'][0], self.ranges['gs_y'][0]])
        ub.extend([self.ranges['gs_x'][1], self.ranges['gs_y'][1]])

        # 遮挡物位置 (遮挡因素) - 3维
        lb.extend([
            self.ranges['obs_x'][0],
            self.ranges['obs_y'][0],
            self.ranges['obs_z'][0]
        ])
        ub.extend([
            self.ranges['obs_x'][1],
            self.ranges['obs_y'][1],
            self.ranges['obs_z'][1]
        ])

        # 干扰源参数 (干扰因素) - 6维
        lb.extend([
            self.ranges['int_x'][0],
            self.ranges['int_y'][0],
            self.ranges['int_z'][0],
            self.ranges['int_freq'][0],
            self.ranges['int_bandwidth'][0],
            self.ranges['int_power'][0]
        ])
        ub.extend([
            self.ranges['int_x'][1],
            self.ranges['int_y'][1],
            self.ranges['int_z'][1],
            self.ranges['int_freq'][1],
            self.ranges['int_bandwidth'][1],
            self.ranges['int_power'][1]
        ])

        # 无人机速度 (速度因素)
        for _ in range(self.num_uavs):
            lb.append(self.ranges['uav_velocities'][0])
            ub.append(self.ranges['uav_velocities'][1])

        return lb, ub

    def optimize(self):
        """执行综合优化 - 寻找使通信劣化程度最大的极限组合干扰场景"""
        print("=" * 70)
        print("开始综合优化 - 寻找极限组合干扰场景")
        print("=" * 70)
        print(f"无人机数量: {self.num_uavs}")
        print(f"决策变量维度: {self.total_dim}")
        print(f"种群大小: {self.population_size}")
        print(f"最大代数: {self.max_generations}")
        print(f"综合权重配置:")
        for factor, weight in self.degradation_model.combined_weights.items():
            print(f"  {factor}: {weight:.2f}")

        # 定义优化问题
        class CombinedProblem(ea.Problem):
            def __init__(self, parent):
                self.parent = parent
                name = 'CombinedOptimization3D'
                M = 1  # 目标维度
                maxormins = [-1]  # 最大化目标 (最大化劣化程度)
                Dim = self.parent.total_dim  # 决策变量维度
                varTypes = [0] * Dim  # 连续型变量
                lb = self.parent.lb
                ub = self.parent.ub
                lbin = [1] * Dim  # 下边界包含
                ubin = [1] * Dim  # 上边界包含

                ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

            def evalVars(self, Vars):
                ObjV = np.zeros((Vars.shape[0], 1))
                for i in range(Vars.shape[0]):
                    total_degradation, _ = self.parent.degradation_model.calculate_combined_degradation(Vars[i, :])
                    ObjV[i, 0] = total_degradation
                return ObjV

        # 实例化问题对象
        problem = CombinedProblem(self)

        # 构建算法
        algorithm = ea.soea_SEGA_templet(
            problem,
            ea.Population(Encoding='RI', NIND=self.population_size),
            MAXGEN=self.max_generations,
            logTras=10,
            trappedValue=1e-6,
            maxTrappedCount=10
        )

        # 求解
        print("\n开始遗传算法优化...")
        res = ea.optimize(algorithm, verbose=True, drawing=1, outputMsg=True)

        # 提取最优解
        best_solution_vars = res['Vars'][0]
        best_fitness = res['ObjV'][0][0]

        # 计算各分量劣化程度
        total_degradation, components = self.degradation_model.calculate_combined_degradation(best_solution_vars)

        # 解析最优解
        best_solution = self._parse_solution(best_solution_vars, total_degradation, components)

        print("\n" + "=" * 70)
        print("综合优化完成!")
        print("=" * 70)

        return best_solution, best_fitness, self.optimization_history

    def _parse_solution(self, decision_vars, total_degradation, components):
        """解析决策变量为可读的解决方案 - 三维版本"""
        # 解析各个变量
        gs_position = decision_vars[0:2]  # 地面站位置
        obs_position = decision_vars[2:5]  # 遮挡物位置 (三维)
        int_params = decision_vars[5:11]  # 干扰源参数 (三维)
        uav_velocities = decision_vars[11:11 + self.num_uavs]  # 无人机速度

        return {
            'gs_position': gs_position,
            'obstruction_position': obs_position,  # 现在是三维
            'interference_params': {
                'position': int_params[0:3],  # 三维位置
                'frequency': int_params[3],
                'bandwidth': int_params[4],
                'power': int_params[5]
            },
            'uav_velocities': uav_velocities,
            'total_degradation': total_degradation,
            'distance_degradation': components['distance'],
            'obstruction_degradation': components['obstruction'],
            'interference_degradation': components['interference'],
            'velocity_degradation': components['velocity'],
            'contributions': components['contributions']
        }

    def analyze_results(self, best_solution, best_fitness):
        """分析并显示优化结果 - 三维版本"""
        print("\n" + "=" * 70)
        print("极限组合干扰场景分析结果")
        print("=" * 70)

        print(f"\n🎯 总通信劣化程度: {best_fitness:.4f}")

        print(f"\n📊 各因素贡献度:")
        print(f"  距离因素 D₁(d): {best_solution['distance_degradation']:.4f} "
              f"(权重: {self.degradation_model.combined_weights['distance']:.2f})")
        print(f"  遮挡因素 D₂(o): {best_solution['obstruction_degradation']:.4f} "
              f"(权重: {self.degradation_model.combined_weights['obstruction']:.2f})")
        print(f"  干扰因素 D₃(i): {best_solution['interference_degradation']:.4f} "
              f"(权重: {self.degradation_model.combined_weights['interference']:.2f})")
        print(f"  速度因素 D₄(v): {best_solution['velocity_degradation']:.4f} "
              f"(权重: {self.degradation_model.combined_weights['velocity']:.2f})")

        print(f"\n📍 最优参数组合:")
        print(f"  地面站位置: ({best_solution['gs_position'][0]:.2f}, {best_solution['gs_position'][1]:.2f})")
        print(f"  遮挡物位置: ({best_solution['obstruction_position'][0]:.2f}, "
              f"{best_solution['obstruction_position'][1]:.2f}, {best_solution['obstruction_position'][2]:.2f})")
        print(f"  干扰源位置: ({best_solution['interference_params']['position'][0]:.2f}, "
              f"{best_solution['interference_params']['position'][1]:.2f}, "
              f"{best_solution['interference_params']['position'][2]:.2f})")
        print(f"  干扰频率: {best_solution['interference_params']['frequency'] / 1e6:.2f} MHz")
        print(f"  干扰带宽: {best_solution['interference_params']['bandwidth'] / 1e6:.2f} MHz")
        print(f"  干扰功率: {best_solution['interference_params']['power']:.2f} W")

        print(f"\n🚁 无人机最优速度:")
        for i, velocity in enumerate(best_solution['uav_velocities']):
            print(f"  无人机{i + 1}: {velocity:.2f} m/s")

        # 显示无人机高度信息
        print(f"\n📏 无人机高度分布:")
        for i, uav_pos in enumerate(self.uav_positions):
            print(f"  无人机{i + 1}: {uav_pos[2]:.1f}米")

        print(f"\n💡 分析建议:")
        max_contrib_factor = max(self.degradation_model.combined_weights,
                                 key=lambda x: self.degradation_model.combined_weights[x] *
                                               best_solution[f'{x}_degradation'])
        print(f"  主要劣化因素: {max_contrib_factor}")

        if max_contrib_factor == 'distance':
            print("  建议优化地面站位置或无人机分布")
        elif max_contrib_factor == 'obstruction':
            print("  建议避开遮挡区域或优化飞行高度")
        elif max_contrib_factor == 'interference':
            print("  建议使用抗干扰技术或频段切换")
        else:
            print("  建议优化飞行速度策略")


# 测试函数
def test_combined_optimizer():
    """测试综合优化器 - 三维版本"""
    # 创建测试无人机位置（三维）
    uav_positions = np.array([
        [100, 200, 50],  # 无人机1: (x, y, z)
        [150, 180, 60],  # 无人机2
        [200, 220, 55],  # 无人机3
        [180, 150, 65]  # 无人机4
    ])

    print("测试三维综合优化器...")
    optimizer = CombinedOptimizer(
        uav_positions=uav_positions,
        population_size=50,  # 测试时使用较小种群
        max_generations=100  # 测试时使用较少代数
    )

    # 执行优化
    best_solution, best_fitness, history = optimizer.optimize()

    # 分析结果
    optimizer.analyze_results(best_solution, best_fitness)

    return best_solution, best_fitness


if __name__ == "__main__":
    test_combined_optimizer()