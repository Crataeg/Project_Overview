import numpy as np
import geatpy as ea
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class ObstructionDegradationModel:
    """
    遮挡相关的通信劣化程度计算模型
    计算遮挡程度对无人机通信性能的影响（三维版本）
    """

    def __init__(self):
        """初始化遮挡劣化模型参数"""
        # 遮挡相关参数
        self.obstruction_params = {
            'PLR0': 0.01,  # 基础丢包率
            'P_ho0': 0.05,  # 基础切换失败率
            'max_obstruction': 1.0,  # 最大遮挡程度 (0-1)
            'obstruction_outage_threshold': 0.8  # 遮挡中断阈值
        }

        # 权重系数 - 用户可调整 (β系数)
        self.beta_weights = {
            'MP': 0.15,  # 多径损耗权重 β₁
            'PLR': 0.15,  # 丢包率权重 β₂
            'SNR_loss': 0.15,  # 信噪比损失权重 β₃
            'P_outage': 0.10,  # 中断概率权重 β₄
            'Delay': 0.10,  # 延迟权重 β₅
            'Throughput_loss': 0.10,  # 吞吐量损失权重 β₆
            'Handover_failure': 0.10,  # 切换失败权重 β₇
            'Multi_hop_outage': 0.05,  # 多跳中断权重 β₈
            'System_throughput_loss': 0.05,  # 系统吞吐量损失权重 β₉
            'Collision_rate': 0.05  # 冲突率权重 β₁₀
        }

        # 系数参数 - 用户可调整 (k系数)
        self.coefficients = {
            'k5': 0.8,  # 多径损耗系数 k₅
            'k6': 0.4,  # 丢包率系数 k₆
            'k7': 0.6,  # 信噪比损失系数 k₇
            'k8': 2.0,  # 中断概率系数 k₈
            'k9': 0.3,  # 切换失败系数 k₉
            'k10': 0.2,  # 延迟系数 k₁₀
            'k11': 0.25,  # 吞吐量损失系数 k₁₁
            'k12': 0.15,  # 多跳中断系数 k₁₂
            'k13': 0.1,  # 系统吞吐量损失系数 k₁₃
            'k14': 0.08  # 冲突率系数 k₁₄
        }

    def multipath_loss(self, o: float) -> float:
        """
        计算多径损耗
        MP(o) = k₅·o
        """
        return self.coefficients['k5'] * o

    def packet_loss_rate(self, o: float) -> float:
        """
        计算遮挡丢包率
        PLR(o) = PLR₀ + k₆·o
        """
        return self.obstruction_params['PLR0'] + self.coefficients['k6'] * o

    def snr_loss(self, o: float) -> float:
        """
        计算遮挡信噪比损失
        SNR_loss(o) = k₇·o
        """
        return self.coefficients['k7'] * o

    def outage_probability(self, o: float) -> float:
        """
        计算遮挡中断概率
        P_outage(o) = 1 - exp(-k₈·o)
        """
        return 1 - np.exp(-self.coefficients['k8'] * o)

    def communication_delay(self, o: float) -> float:
        """
        计算通信延迟
        Delay(o) = k₁₀·o
        """
        return self.coefficients['k10'] * o

    def throughput_loss(self, o: float) -> float:
        """
        计算吞吐量损失
        Throughput_loss(o) = k₁₁·o
        """
        return self.coefficients['k11'] * o

    def handover_failure(self, o: float) -> float:
        """
        计算遮挡切换失败率
        Handover_failure(o) = P_ho₀ + k₉·o
        """
        return self.obstruction_params['P_ho0'] + self.coefficients['k9'] * o

    def multi_hop_outage(self, o: float) -> float:
        """
        计算多跳中断概率
        Multi_hop_outage(o) = 1 - (1 - P_outage(o))^N
        简化假设：每跳遮挡程度相同，N=3跳
        """
        single_hop_outage = self.outage_probability(o)
        return 1 - (1 - single_hop_outage) ** 3

    def system_throughput_loss(self, o: float) -> float:
        """
        计算系统吞吐量损失
        System_throughput_loss(o) = Throughput_loss(o) × N_hop
        假设N_hop=3
        """
        return self.throughput_loss(o) * 3

    def collision_rate(self, o: float) -> float:
        """
        计算冲突率
        Collision_rate(o) = k₁₄·o
        """
        return self.coefficients['k14'] * o

    def calculate_single_uav_degradation(self, o: float) -> float:
        """
        计算单个无人机的遮挡通信劣化程度
        D₂(o) = β₁ × MP(o) + β₂ × PLR(o) + ... + β₁₀ × Collision_rate(o)
        """
        # 归一化各分量到0-1范围
        components = {
            'MP': min(self.multipath_loss(o) / 2.0, 1.0),  # 假设最大多径损耗2.0
            'PLR': min(self.packet_loss_rate(o), 1.0),
            'SNR_loss': min(self.snr_loss(o), 1.0),
            'P_outage': self.outage_probability(o),
            'Delay': min(self.communication_delay(o) / 5.0, 1.0),  # 假设最大延迟5.0
            'Throughput_loss': min(self.throughput_loss(o), 1.0),
            'Handover_failure': min(self.handover_failure(o), 1.0),
            'Multi_hop_outage': self.multi_hop_outage(o),
            'System_throughput_loss': min(self.system_throughput_loss(o), 1.0),
            'Collision_rate': min(self.collision_rate(o), 1.0)
        }

        # 加权求和
        total_degradation = 0
        for component, weight in self.beta_weights.items():
            total_degradation += weight * components[component]

        return min(total_degradation, 1.0)  # 确保在0-1范围内


class ObstructionEffectModel3D:
    """
    三维遮挡效应模型
    计算遮挡物位置对各个无人机的遮挡程度影响，考虑高度因素
    """

    def __init__(self, uav_positions: List[Tuple[float, float, float]]):
        """
        初始化三维遮挡效应模型

        Args:
            uav_positions: 无人机位置列表 [(x, y, z), ...]
        """
        self.uav_positions = np.array(uav_positions)

    def calculate_obstruction_effect(self,
                                     obstruction_pos: Tuple[float, float, float],
                                     obstruction_height: float = 50.0,
                                     obstruction_strength: float = 1.0,
                                     horizontal_range: float = 300.0,
                                     vertical_range: float = 150.0) -> List[float]:
        """
        计算遮挡物对各个无人机的遮挡程度（三维版本）

        Args:
            obstruction_pos: 遮挡物位置 (x, y, z)
            obstruction_height: 遮挡物高度
            obstruction_strength: 遮挡物强度 (0-1)
            horizontal_range: 水平作用范围 (米)
            vertical_range: 垂直作用范围 (米)

        Returns:
            每个无人机的遮挡程度列表
        """
        obstruction_array = np.array(obstruction_pos)
        obstruction_levels = []

        for uav_pos in self.uav_positions:
            # 计算水平距离
            horizontal_distance = np.linalg.norm(uav_pos[:2] - obstruction_array[:2])

            # 计算高度差
            height_diff = abs(uav_pos[2] - obstruction_array[2])

            # 计算三维遮挡效应
            if horizontal_distance <= horizontal_range and height_diff <= vertical_range:
                # 水平方向高斯衰减
                horizontal_effect = np.exp(-0.5 * (horizontal_distance / (horizontal_range / 3)) ** 2)

                # 垂直方向高斯衰减
                vertical_effect = np.exp(-0.5 * (height_diff / (vertical_range / 3)) ** 2)

                # 考虑遮挡物高度的影响
                height_factor = self._calculate_height_factor(uav_pos[2], obstruction_array[2], obstruction_height)

                # 综合遮挡程度
                obstruction_level = obstruction_strength * horizontal_effect * vertical_effect * height_factor
            else:
                obstruction_level = 0.0

            obstruction_levels.append(min(obstruction_level, 1.0))

        return obstruction_levels

    def _calculate_height_factor(self, uav_height: float, obstruction_height: float,
                                 obstruction_obj_height: float) -> float:
        """
        计算高度因子
        考虑遮挡物高度对遮挡效果的影响

        Args:
            uav_height: 无人机高度
            obstruction_height: 遮挡物基座高度
            obstruction_obj_height: 遮挡物自身高度

        Returns:
            高度因子 (0-1)
        """
        total_obstruction_height = obstruction_height + obstruction_obj_height

        # 如果无人机在遮挡物下方，遮挡效果较强
        if uav_height <= total_obstruction_height:
            return 1.0
        else:
            # 无人机高于遮挡物时，遮挡效果随高度差衰减
            height_diff = uav_height - total_obstruction_height
            decay_factor = np.exp(-height_diff / 50.0)  # 衰减系数，可调整
            return max(decay_factor, 0.1)  # 保持最小遮挡效果

    def calculate_line_of_sight(self,
                                uav_pos: Tuple[float, float, float],
                                obstruction_pos: Tuple[float, float, float],
                                obstruction_height: float = 50.0) -> bool:
        """
        计算无人机与遮挡物之间是否存在视线遮挡

        Args:
            uav_pos: 无人机位置 (x, y, z)
            obstruction_pos: 遮挡物位置 (x, y, z)
            obstruction_height: 遮挡物高度

        Returns:
            True表示存在视线遮挡，False表示无遮挡
        """
        # 简化的视线判断：如果无人机高度低于遮挡物总高度，则存在遮挡
        total_obstruction_height = obstruction_pos[2] + obstruction_height
        return uav_pos[2] <= total_obstruction_height


class ObstructionOptimizationProblem3D(ea.Problem):
    """
    三维遮挡物位置优化问题定义
    遗传算法寻找使多无人机通信劣化程度最大的遮挡物三维位置
    """

    def __init__(self, uav_positions: List[Tuple[float, float, float]],
                 degradation_model: ObstructionDegradationModel,
                 obstruction_model: ObstructionEffectModel3D,
                 search_area: Tuple[float, float, float, float, float, float] = (0, 0, 0, 1000, 1000, 200),
                 obstruction_strength: float = 1.0,
                 obstruction_height: float = 50.0):
        """
        初始化三维优化问题

        Args:
            uav_positions: 无人机位置列表 [(x1,y1,z1), (x2,y2,z2), ...]
            degradation_model: 通信劣化程度计算模型
            obstruction_model: 三维遮挡效应模型
            search_area: 搜索区域 (x_min, y_min, z_min, x_max, y_max, z_max)
            obstruction_strength: 遮挡物强度
            obstruction_height: 遮挡物高度
        """
        self.uav_positions = np.array(uav_positions)
        self.degradation_model = degradation_model
        self.obstruction_model = obstruction_model
        self.search_area = search_area
        self.obstruction_strength = obstruction_strength
        self.obstruction_height = obstruction_height

        # 问题定义
        name = '3D_Obstruction_Position_Optimization'  # 问题名称
        M = 1  # 目标维度 (单目标优化)
        maxormins = [-1]  # 目标最小最大化标记 (-1表示最大化，1表示最小化)
        Dim = 3  # 决策变量维度 (遮挡物的x,y,z坐标)

        # 变量类型: 0-连续, 1-离散
        varTypes = [0, 0, 0]  # 三个连续变量

        # 决策变量范围
        lb = [search_area[0], search_area[1], search_area[2]]  # 下界
        ub = [search_area[3], search_area[4], search_area[5]]  # 上界

        # 边界包含性 (1-包含边界，0-不包含)
        lbin = [1, 1, 1]  # 包含下界
        ubin = [1, 1, 1]  # 包含上界

        # 调用父类构造函数初始化问题
        super().__init__(name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        """
        目标函数 - 计算种群中每个个体的适应度

        Args:
            pop: 种群对象，包含决策变量矩阵 pop.Phen
        """
        # 获取决策变量矩阵，每一行代表一个遮挡物位置 (x,y,z)
        x = pop.Phen  # 形状: (种群大小, 3)

        # 初始化适应度数组
        f_values = np.zeros((x.shape[0], 1))

        # 遍历种群中的每个个体
        for i in range(x.shape[0]):
            # 当前个体的遮挡物位置
            obstruction_pos = (x[i, 0], x[i, 1], x[i, 2])

            # 计算遮挡物对各个无人机的遮挡程度
            obstruction_levels = self.obstruction_model.calculate_obstruction_effect(
                obstruction_pos, self.obstruction_height, self.obstruction_strength)

            # 计算总通信劣化程度
            total_degradation = 0
            num_uavs = len(self.uav_positions)

            # 对每个无人机计算遮挡劣化程度并求和
            for j, obstruction_level in enumerate(obstruction_levels):
                # 计算该遮挡程度下的通信劣化程度
                uav_degradation = self.degradation_model.calculate_single_uav_degradation(obstruction_level)

                # 累加到总劣化程度
                total_degradation += uav_degradation

            # 计算平均劣化程度作为适应度值
            avg_degradation = total_degradation / num_uavs if num_uavs > 0 else 0

            # 存储适应度值 (由于设置了maxormins=-1，这里直接存储，算法会自动处理最大化)
            f_values[i, 0] = avg_degradation

        # 设置种群适应度
        pop.ObjV = f_values


class ObstructionOptimizer3D:
    """
    三维遮挡物位置优化器
    使用遗传算法寻找最优遮挡物三维位置
    """

    def __init__(self,
                 uav_positions: List[Tuple[float, float, float]],
                 population_size: int = 50,
                 max_generations: int = 100,
                 obstruction_strength: float = 1.0,
                 obstruction_height: float = 50.0):
        """
        初始化三维优化器

        Args:
            uav_positions: 无人机位置列表 [(x, y, z), ...]
            population_size: 种群大小
            max_generations: 最大进化代数
            obstruction_strength: 遮挡物强度
            obstruction_height: 遮挡物高度
        """
        self.uav_positions = uav_positions
        self.population_size = population_size
        self.max_generations = max_generations
        self.obstruction_strength = obstruction_strength
        self.obstruction_height = obstruction_height

        # 创建通信劣化模型
        self.degradation_model = ObstructionDegradationModel()

        # 创建三维遮挡效应模型
        self.obstruction_model = ObstructionEffectModel3D(uav_positions)

        # 计算搜索区域 (基于无人机位置自动确定)
        uav_array = np.array(uav_positions)
        x_min, y_min, z_min = np.min(uav_array, axis=0)
        x_max, y_max, z_max = np.max(uav_array, axis=0)

        # 扩展搜索区域，给遮挡物更多选择空间
        horizontal_padding = max(x_max - x_min, y_max - y_min) * 0.5
        vertical_padding = (z_max - z_min) * 0.5

        search_area = (x_min - horizontal_padding, y_min - horizontal_padding, max(0, z_min - vertical_padding),
                       x_max + horizontal_padding, y_max + horizontal_padding, z_max + vertical_padding)

        # 创建三维优化问题
        self.problem = ObstructionOptimizationProblem3D(
            uav_positions, self.degradation_model, self.obstruction_model,
            search_area, obstruction_strength, obstruction_height)

        # 配置遗传算法
        self.algorithm = ea.soea_SEGA_templet(
            self.problem,
            ea.Population(Encoding='RI', NIND=population_size),
            MAXGEN=max_generations,
            logTras=10,  # 每隔10代记录日志
            trappedValue=1e-6,  # 收敛阈值
            maxTrappedCount=10  # 最大停滞代数
        )

        # 存储优化结果
        self.optimization_result = None

    def run_optimization(self) -> Dict[str, Any]:
        """
        运行三维遮挡物位置优化

        Returns:
            优化结果字典
        """
        print("开始三维遮挡物位置优化...")
        print(f"无人机数量: {len(self.uav_positions)}")
        print(f"遮挡物强度: {self.obstruction_strength}")
        print(f"遮挡物高度: {self.obstruction_height}")
        print(f"搜索区域: {self.problem.search_area}")

        # 运行遗传算法
        [BestIndi, population] = self.algorithm.run()

        # 提取最优解
        best_position = BestIndi.Phen[0]  # 最优遮挡物位置 [x, y, z]
        best_fitness = float(BestIndi.ObjV[0])  # 最优适应度值

        # 计算最优遮挡物位置对各个无人机的遮挡程度
        obstruction_levels = self.obstruction_model.calculate_obstruction_effect(
            (best_position[0], best_position[1], best_position[2]),
            self.obstruction_height, self.obstruction_strength)

        # 计算各无人机的具体劣化程度和视线信息
        uav_degradations = []
        for i, (uav_pos, obstruction_level) in enumerate(zip(self.uav_positions, obstruction_levels)):
            degradation = self.degradation_model.calculate_single_uav_degradation(obstruction_level)
            distance = np.linalg.norm(np.array(uav_pos) - best_position)

            # 计算视线遮挡情况
            los_blocked = self.obstruction_model.calculate_line_of_sight(
                tuple(uav_pos), tuple(best_position), self.obstruction_height)

            uav_degradations.append({
                'uav_id': i + 1,
                'position': uav_pos,
                'distance_to_obstruction': distance,
                'obstruction_level': obstruction_level,
                'degradation': degradation,
                'line_of_sight_blocked': los_blocked
            })

        # 构建结果字典
        self.optimization_result = {
            'best_obstruction_position': best_position.tolist(),
            'best_fitness': best_fitness,
            'obstruction_strength': self.obstruction_strength,
            'obstruction_height': self.obstruction_height,
            'uav_degradations': uav_degradations,
            'average_obstruction_level': np.mean(obstruction_levels),
            'max_obstruction_level': np.max(obstruction_levels),
            'los_blocked_count': sum(1 for uav in uav_degradations if uav['line_of_sight_blocked']),
            'search_area': self.problem.search_area,
            'optimization_parameters': {
                'population_size': self.population_size,
                'max_generations': self.max_generations
            }
        }

        print(f"优化完成! 最优遮挡物位置: ({best_position[0]:.2f}, {best_position[1]:.2f}, {best_position[2]:.2f})")
        print(f"平均遮挡程度: {np.mean(obstruction_levels):.4f}")
        print(f"最大遮挡程度: {np.max(obstruction_levels):.4f}")
        print(f"平均通信劣化程度: {best_fitness:.4f}")
        print(f"视线遮挡无人机数量: {self.optimization_result['los_blocked_count']}/{len(uav_degradations)}")

        return self.optimization_result

    def plot_optimization_results(self):
        """
        绘制三维优化结果可视化图表
        """
        if self.optimization_result is None:
            print("请先运行优化!")
            return

        fig = plt.figure(figsize=(18, 12))

        # 创建三维散点图
        ax1 = fig.add_subplot(231, projection='3d')
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)

        # 提取数据
        uav_positions = np.array(self.uav_positions)
        best_obstruction = self.optimization_result['best_obstruction_position']
        degradations = [item['degradation'] for item in self.optimization_result['uav_degradations']]
        obstruction_levels = [item['obstruction_level'] for item in self.optimization_result['uav_degradations']]
        los_blocked = [item['line_of_sight_blocked'] for item in self.optimization_result['uav_degradations']]

        # 图1: 三维位置分布
        scatter1 = ax1.scatter(uav_positions[:, 0], uav_positions[:, 1], uav_positions[:, 2],
                               c=obstruction_levels, cmap='hot', s=100,
                               label='无人机位置', alpha=0.8, depthshade=True)

        # 绘制最优遮挡物位置
        ax1.scatter(best_obstruction[0], best_obstruction[1], best_obstruction[2],
                    c='red', marker='X', s=200, label='最优遮挡物',
                    edgecolors='white', linewidth=2)

        # 绘制遮挡物高度柱体
        ax1.plot([best_obstruction[0], best_obstruction[0]],
                 [best_obstruction[1], best_obstruction[1]],
                 [best_obstruction[2], best_obstruction[2] + self.obstruction_height],
                 'red', linewidth=3, label='遮挡物高度')

        # 绘制连接线（视线）
        for i, uav_pos in enumerate(uav_positions):
            color = 'red' if los_blocked[i] else 'green'
            linestyle = '-' if los_blocked[i] else '--'
            ax1.plot([uav_pos[0], best_obstruction[0]],
                     [uav_pos[1], best_obstruction[1]],
                     [uav_pos[2], best_obstruction[2]],
                     color=color, linestyle=linestyle, alpha=0.5)

        ax1.set_xlabel('X坐标 (米)')
        ax1.set_ylabel('Y坐标 (米)')
        ax1.set_zlabel('Z坐标 (米)')
        ax1.set_title('三维位置分布和视线分析')
        ax1.legend()
        plt.colorbar(scatter1, ax=ax1, label='遮挡程度')

        # 图2: 遮挡程度与通信劣化关系
        colors = ['red' if blocked else 'green' for blocked in los_blocked]
        scatter2 = ax2.scatter(obstruction_levels, degradations, c=colors, s=100, alpha=0.7)
        ax2.set_xlabel('遮挡程度')
        ax2.set_ylabel('通信劣化程度')
        ax2.set_title('遮挡程度 vs 通信劣化程度\n(红色:视线遮挡, 绿色:无遮挡)')
        ax2.grid(True, alpha=0.3)

        # 图3: 各无人机通信劣化程度
        uav_ids = [item['uav_id'] for item in self.optimization_result['uav_degradations']]
        bars = ax3.bar(uav_ids, degradations, color=['red' if blocked else 'skyblue'
                                                     for blocked in los_blocked], alpha=0.7)

        for bar, degradation in zip(bars, degradations):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{degradation:.3f}', ha='center', va='bottom', fontsize=9)

        ax3.set_xlabel('无人机编号')
        ax3.set_ylabel('通信劣化程度')
        ax3.set_title('各无人机通信劣化程度\n(红色:视线遮挡)')
        ax3.set_xticks(uav_ids)
        ax3.grid(True, alpha=0.3)

        # 图4: 遮挡程度分布
        ax4.bar(uav_ids, obstruction_levels, color=['red' if blocked else 'orange'
                                                    for blocked in los_blocked], alpha=0.7)
        ax4.set_xlabel('无人机编号')
        ax4.set_ylabel('遮挡程度')
        ax4.set_title('各无人机遮挡程度\n(红色:视线遮挡)')
        ax4.set_xticks(uav_ids)
        ax4.grid(True, alpha=0.3)

        # 图5: 高度分布
        uav_heights = [pos[2] for pos in uav_positions]
        ax5.bar(uav_ids, uav_heights, color=['red' if blocked else 'lightgreen'
                                             for blocked in los_blocked], alpha=0.7)
        ax5.axhline(y=best_obstruction[2] + self.obstruction_height, color='red',
                    linestyle='--', label='遮挡物顶部高度')
        ax5.axhline(y=best_obstruction[2], color='orange',
                    linestyle='--', label='遮挡物基座高度')
        ax5.set_xlabel('无人机编号')
        ax5.set_ylabel('高度 (米)')
        ax5.set_title('无人机高度分布')
        ax5.set_xticks(uav_ids)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_optimization_summary(self) -> str:
        """
        获取优化结果摘要

        Returns:
            优化结果摘要字符串
        """
        if self.optimization_result is None:
            return "请先运行优化!"

        result = self.optimization_result
        summary = f"""
=== 三维遮挡优化结果摘要 ===
最优遮挡物位置: ({result['best_obstruction_position'][0]:.2f}, {result['best_obstruction_position'][1]:.2f}, {result['best_obstruction_position'][2]:.2f})
遮挡物高度: {result['obstruction_height']:.1f}m
平均通信劣化程度: {result['best_fitness']:.4f}
遮挡物强度: {result['obstruction_strength']:.2f}
平均遮挡程度: {result['average_obstruction_level']:.4f}
最大遮挡程度: {result['max_obstruction_level']:.4f}
视线遮挡无人机数量: {result['los_blocked_count']}/{len(result['uav_degradations'])}

各无人机详情:
"""
        for uav_info in result['uav_degradations']:
            los_status = "视线遮挡" if uav_info['line_of_sight_blocked'] else "无遮挡"
            summary += f"无人机 {uav_info['uav_id']}: 位置{uav_info['position']}, 高度{uav_info['position'][2]:.1f}m, "
            summary += f"距离遮挡物{uav_info['distance_to_obstruction']:.1f}m, "
            summary += f"遮挡程度{uav_info['obstruction_level']:.4f}, 劣化程度{uav_info['degradation']:.4f}, {los_status}\n"

        return summary


# 使用示例和测试函数
def main():
    """
    主函数 - 演示如何使用三维遮挡优化中间件
    """
    # 1. 生成模拟无人机位置 (在500x500x200三维区域内分布)
    np.random.seed(42)  # 设置随机种子保证可重复性
    num_uavs = 8
    uav_positions = []

    # 在区域中心生成无人机群，不同高度
    center_x, center_y = 500, 500
    base_height = 100
    radius = 200

    for i in range(num_uavs):
        angle = 2 * np.pi * i / num_uavs
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        z = base_height + np.random.uniform(-20, 50)  # 高度变化
        uav_positions.append((x, y, z))

    print("生成的无人机位置:")
    for i, pos in enumerate(uav_positions):
        print(f"无人机 {i + 1}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    # 2. 创建三维优化器并运行优化
    optimizer = ObstructionOptimizer3D(
        uav_positions=uav_positions,
        population_size=30,  # 较小的种群大小用于快速演示
        max_generations=50,  # 较少的代数用于快速演示
        obstruction_strength=0.9,  # 遮挡物强度
        obstruction_height=60.0  # 遮挡物高度
    )

    # 3. 运行优化
    result = optimizer.run_optimization()

    # 4. 显示详细结果
    print("\n" + "=" * 50)
    print(optimizer.get_optimization_summary())

    # 5. 绘制结果图表
    optimizer.plot_optimization_results()

    return result


if __name__ == "__main__":
    # 运行演示
    result = main()