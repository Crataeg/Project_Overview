import numpy as np
import geatpy as ea
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class DistanceDegradationModel:
    """
    距离和高度相关的通信劣化程度计算模型
    计算单个无人机到地面站距离和高度对通信性能的影响
    """

    def __init__(self):
        """初始化距离和高度的劣化模型参数"""
        # 距离和高度的相关参数
        self.distance_params = {
            'PL0': 40,  # 参考距离处的路径损耗 (dB)
            'd0': 100,  # 参考距离 (米)
            'n': 2.5,  # 路径损耗指数
            'd_max': 2000,  # 最大通信距离 (米)
            'd_out': 1500,  # 中断距离 (米)
            'PLR0': 0.01,  # 基础丢包率
            'SNR0': 30,  # 参考信噪比 (dB)
            'Delay0': 10,  # 基础延迟 (ms)
            'P_ho0': 0.05,  # 基础切换失败率
            'CR0': 0.02,  # 基础冲突率
            'N_hop': 3,  # 多跳数量
            'h_max': 500,  # 最大允许高度 (米)
            'h_optimal': 150,  # 最优通信高度 (米)
            'h_penalty_threshold': 300  # 高度惩罚阈值 (米)
        }

        # 权重系数 - 用户可调整
        self.alpha_weights = {
            'PL': 0.12,  # 路径损耗权重
            'PLR': 0.12,  # 丢包率权重
            'SNR_loss': 0.12,  # 信噪比损失权重
            'P_outage': 0.10,  # 中断概率权重
            'Delay': 0.10,  # 延迟权重
            'Throughput_loss': 0.10,  # 吞吐量损失权重
            'Handover_failure': 0.08,  # 切换失败权重
            'Multi_hop_outage': 0.05,  # 多跳中断权重
            'System_throughput_loss': 0.05,  # 系统吞吐量损失权重
            'Collision_rate': 0.05,  # 冲突率权重
            'Height_penalty': 0.11  # 高度惩罚权重
        }

        # 系数参数 - 用户可调整
        self.coefficients = {
            'k1': 0.3,  # 丢包率距离系数
            'k2': 0.001,  # 延迟距离系数 (ms/米)
            'k3': 0.0001,  # 切换失败距离系数
            'k4': 0.00005,  # 冲突率距离系数
            'k5': 0.002  # 高度惩罚系数
        }

    def calculate_3d_distance(self, uav_pos: Tuple[float, float, float],
                              gs_pos: Tuple[float, float, float]) -> float:
        """
        计算三维空间中的距离
        """
        dx = uav_pos[0] - gs_pos[0]
        dy = uav_pos[1] - gs_pos[1]
        dz = uav_pos[2] - gs_pos[2]
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def height_penalty(self, uav_height: float) -> float:
        """
        计算高度惩罚因子
        在最优高度附近惩罚最小，随着偏离最优高度惩罚增加
        """
        h_optimal = self.distance_params['h_optimal']
        h_max = self.distance_params['h_max']
        h_penalty_threshold = self.distance_params['h_penalty_threshold']

        # 高度偏离最优值的绝对值
        height_deviation = abs(uav_height - h_optimal)

        if height_deviation <= 50:  # 在最优高度±50米内，惩罚很小
            penalty = 0.05 * (height_deviation / 50)
        elif height_deviation <= h_penalty_threshold:
            penalty = 0.05 + 0.3 * ((height_deviation - 50) / (h_penalty_threshold - 50))
        else:
            penalty = 0.35 + 0.6 * ((height_deviation - h_penalty_threshold) / (h_max - h_penalty_threshold))

        return min(penalty, 1.0)

    def path_loss(self, d: float, h: float = 0) -> float:
        """
        计算路径损耗，考虑高度影响
        PL(d) = PL0 + 10n·log10(d/d0) + ΔPL(h)
        """
        if d <= 0:
            return float('inf')  # 距离为0时路径损耗无穷大

        # 基础路径损耗
        base_pl = self.distance_params['PL0'] + 10 * self.distance_params['n'] * np.log10(
            d / self.distance_params['d0'])

        # 高度修正因子 (高度增加可能导致信号衰减)
        height_factor = 0.1 * abs(h - self.distance_params['h_optimal']) / 100

        return base_pl + height_factor

    def packet_loss_rate(self, d: float, h: float = 0) -> float:
        """
        计算丢包率，考虑高度影响
        PLR(d) = PLR0 + k1·(d/d_max)² + ΔPLR(h)
        """
        base_plr = self.distance_params['PLR0'] + self.coefficients['k1'] * (d / self.distance_params['d_max']) ** 2

        # 高度对丢包率的影响
        height_impact = 0.02 * self.height_penalty(h)

        return min(base_plr + height_impact, 1.0)

    def snr_loss(self, d: float, h: float = 0) -> float:
        """
        计算信噪比损失，考虑高度影响
        SNR_loss(d) = SNR0 - PL(d,h)
        """
        return self.distance_params['SNR0'] - self.path_loss(d, h)

    def outage_probability(self, d: float, h: float = 0) -> float:
        """
        计算中断概率，考虑高度影响
        P_outage(d) = 1 - exp(-(d/d_out)²) + ΔP_outage(h)
        """
        base_outage = 1 - np.exp(-(d / self.distance_params['d_out']) ** 2)

        # 高度对中断概率的影响
        height_impact = 0.1 * self.height_penalty(h)

        return min(base_outage + height_impact, 1.0)

    def communication_delay(self, d: float, h: float = 0) -> float:
        """
        计算通信延迟，考虑高度影响
        Delay(d) = Delay0 + k2·d + ΔDelay(h)
        """
        base_delay = self.distance_params['Delay0'] + self.coefficients['k2'] * d

        # 高度对延迟的影响 (高度变化可能导致信号路径变化)
        height_impact = 2 * self.height_penalty(h)

        return base_delay + height_impact

    def throughput_loss(self, d: float, h: float = 0) -> float:
        """
        计算吞吐量损失，考虑高度影响
        Throughput_loss(d) = 1 - (d_max - d)/d_max + ΔThroughput_loss(h)
        """
        base_throughput_loss = 1 - (self.distance_params['d_max'] - d) / self.distance_params['d_max']

        # 高度对吞吐量的影响
        height_impact = 0.1 * self.height_penalty(h)

        return min(base_throughput_loss + height_impact, 1.0)

    def handover_failure(self, d: float, h: float = 0) -> float:
        """
        计算切换失败率，考虑高度影响
        Handover_failure(d) = P_ho0 + k3·d + ΔHandover_failure(h)
        """
        base_handover = self.distance_params['P_ho0'] + self.coefficients['k3'] * d

        # 高度对切换失败率的影响
        height_impact = 0.05 * self.height_penalty(h)

        return min(base_handover + height_impact, 1.0)

    def multi_hop_outage(self, d: float, h: float = 0) -> float:
        """
        计算多跳中断概率，考虑高度影响
        Multi_hop_outage(d) = 1 - ∏(1 - P_outage(d_i))
        简化假设：每跳距离相等
        """
        d_per_hop = d / self.distance_params['N_hop']
        single_hop_outage = self.outage_probability(d_per_hop, h)
        return 1 - (1 - single_hop_outage) ** self.distance_params['N_hop']

    def system_throughput_loss(self, d: float, h: float = 0) -> float:
        """
        计算系统吞吐量损失，考虑高度影响
        System_throughput_loss(d) = Throughput_loss(d) × N_hop + ΔSystem_throughput_loss(h)
        """
        base_system_loss = self.throughput_loss(d, h) * self.distance_params['N_hop']

        # 高度对系统吞吐量的影响
        height_impact = 0.1 * self.height_penalty(h)

        return min(base_system_loss + height_impact, 1.0)

    def collision_rate(self, d: float, h: float = 0) -> float:
        """
        计算冲突率，考虑高度影响
        Collision_rate(d) = CR0 + k4·d + ΔCollision_rate(h)
        """
        base_collision = self.distance_params['CR0'] + self.coefficients['k4'] * d

        # 高度对冲突率的影响
        height_impact = 0.03 * self.height_penalty(h)

        return min(base_collision + height_impact, 1.0)

    def calculate_single_uav_degradation(self, d: float, h: float = 0) -> float:
        """
        计算单个无人机的距离和高度通信劣化程度
        D₁(d,h) = α₁ × PL(d,h) + α₂ × PLR(d,h) + ... + α₁₁ × Height_penalty(h)
        """
        # 归一化各分量到0-1范围
        components = {
            'PL': min(self.path_loss(d, h) / 100, 1.0),  # 假设最大路径损耗100dB
            'PLR': self.packet_loss_rate(d, h),
            'SNR_loss': min(max(self.snr_loss(d, h) / 30, 0), 1.0),  # 信噪比损失归一化
            'P_outage': self.outage_probability(d, h),
            'Delay': min(self.communication_delay(d, h) / 100, 1.0),  # 假设最大延迟100ms
            'Throughput_loss': self.throughput_loss(d, h),
            'Handover_failure': self.handover_failure(d, h),
            'Multi_hop_outage': self.multi_hop_outage(d, h),
            'System_throughput_loss': self.system_throughput_loss(d, h),
            'Collision_rate': self.collision_rate(d, h),
            'Height_penalty': self.height_penalty(h)
        }

        # 加权求和
        total_degradation = 0
        for component, weight in self.alpha_weights.items():
            total_degradation += weight * components[component]

        return min(total_degradation, 1.0)  # 确保在0-1范围内


class GroundStationOptimizationProblem(ea.Problem):
    """
    地面站位置优化问题定义
    遗传算法寻找使多无人机通信劣化程度最大的地面站位置
    """

    def __init__(self, uav_positions: List[Tuple[float, float, float]],
                 degradation_model: DistanceDegradationModel,
                 search_area: Tuple[float, float, float, float] = (0, 0, 1000, 1000),
                 gs_height_range: Tuple[float, float] = (0, 50)):
        """
        初始化优化问题

        Args:
            uav_positions: 无人机位置列表 [(x1,y1,h1), (x2,y2,h2), ...]
            degradation_model: 通信劣化程度计算模型
            search_area: 搜索区域 (x_min, y_min, x_max, y_max)
            gs_height_range: 地面站高度范围 (h_min, h_max)
        """
        self.uav_positions = np.array(uav_positions)  # 转换为numpy数组便于计算
        self.degradation_model = degradation_model
        self.search_area = search_area
        self.gs_height_range = gs_height_range

        # 问题定义
        name = 'Ground_Station_Optimization'  # 问题名称
        M = 1  # 目标维度 (单目标优化)
        maxormins = [-1]  # 目标最小最大化标记 (-1表示最大化，1表示最小化)
        Dim = 3  # 决策变量维度 (地面站的x,y坐标和高度h)

        # 变量类型: 0-连续, 1-离散
        varTypes = [0, 0, 0]  # 三个连续变量

        # 决策变量范围
        lb = [search_area[0], search_area[1], gs_height_range[0]]  # 下界
        ub = [search_area[2], search_area[3], gs_height_range[1]]  # 上界

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
        # 获取决策变量矩阵，每一行代表一个地面站位置 (x,y,h)
        x = pop.Phen  # 形状: (种群大小, 3)

        # 初始化适应度数组
        f_values = np.zeros((x.shape[0], 1))

        # 遍历种群中的每个个体
        for i in range(x.shape[0]):
            # 当前个体的地面站位置
            ground_station_pos = x[i, :]  # [x, y, h]

            # 计算总通信劣化程度
            total_degradation = 0
            num_uavs = len(self.uav_positions)

            # 对每个无人机计算距离劣化程度并求和
            for uav_pos in self.uav_positions:
                # 计算无人机到地面站的三维距离
                distance = np.linalg.norm(uav_pos - ground_station_pos)

                # 获取无人机高度
                uav_height = uav_pos[2]

                # 计算该距离和高度下的通信劣化程度
                uav_degradation = self.degradation_model.calculate_single_uav_degradation(distance, uav_height)

                # 累加到总劣化程度
                total_degradation += uav_degradation

            # 计算平均劣化程度作为适应度值
            avg_degradation = total_degradation / num_uavs if num_uavs > 0 else 0

            # 存储适应度值 (由于设置了maxormins=-1，这里直接存储，算法会自动处理最大化)
            f_values[i, 0] = avg_degradation

        # 设置种群适应度
        pop.ObjV = f_values


class DistanceOptimizer:
    """
    地面站位置优化器
    使用遗传算法寻找最优地面站位置
    """

    def __init__(self,
                 uav_positions: List[Tuple[float, float, float]],
                 population_size: int = 50,
                 max_generations: int = 100,
                 gs_height_range: Tuple[float, float] = (0, 50)):
        """
        初始化优化器

        Args:
            uav_positions: 无人机位置列表 (x, y, h)
            population_size: 种群大小
            max_generations: 最大进化代数
            gs_height_range: 地面站高度范围
        """
        self.uav_positions = uav_positions
        self.population_size = population_size
        self.max_generations = max_generations
        self.gs_height_range = gs_height_range

        # 创建通信劣化模型
        self.degradation_model = DistanceDegradationModel()

        # 计算搜索区域 (基于无人机位置自动确定)
        uav_array = np.array(uav_positions)
        x_min, y_min = np.min(uav_array[:, :2], axis=0)
        x_max, y_max = np.max(uav_array[:, :2], axis=0)

        # 扩展搜索区域，给地面站更多选择空间
        padding = max(x_max - x_min, y_max - y_min) * 0.5
        search_area = (x_min - padding, y_min - padding,
                       x_max + padding, y_max + padding)

        # 创建优化问题
        self.problem = GroundStationOptimizationProblem(
            uav_positions, self.degradation_model, search_area, gs_height_range)

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

    def run_optimization(self) -> Dict:
        """
        运行地面站位置优化

        Returns:
            优化结果字典
        """
        print("开始地面站位置优化...")
        print(f"无人机数量: {len(self.uav_positions)}")
        print(f"搜索区域: {self.problem.search_area}")
        print(f"地面站高度范围: {self.gs_height_range}米")

        # 运行遗传算法
        [BestIndi, population] = self.algorithm.run()

        # 提取最优解
        best_position = BestIndi.Phen[0]  # 最优地面站位置 [x, y, h]
        best_fitness = float(BestIndi.ObjV[0])  # 最优适应度值

        # 计算各无人机的具体劣化程度
        uav_degradations = []
        for uav_pos in self.uav_positions:
            distance = np.linalg.norm(uav_pos - best_position)
            uav_height = uav_pos[2]
            degradation = self.degradation_model.calculate_single_uav_degradation(distance, uav_height)
            uav_degradations.append({
                'position': uav_pos,
                'distance': distance,
                'height': uav_height,
                'degradation': degradation
            })

        # 构建结果字典
        self.optimization_result = {
            'best_ground_station_position': best_position.tolist(),
            'best_fitness': best_fitness,
            'uav_degradations': uav_degradations,
            'search_area': self.problem.search_area,
            'gs_height_range': self.gs_height_range,
            'optimization_parameters': {
                'population_size': self.population_size,
                'max_generations': self.max_generations
            }
        }

        print(f"优化完成! 最优地面站位置: ({best_position[0]:.2f}, {best_position[1]:.2f}, {best_position[2]:.2f})")
        print(f"平均通信劣化程度: {best_fitness:.4f}")

        return self.optimization_result

    def plot_optimization_results(self):
        """
        绘制优化结果可视化图表
        """
        if self.optimization_result is None:
            print("请先运行优化!")
            return

        fig = plt.figure(figsize=(18, 6))

        # 提取数据
        uav_positions = np.array(self.uav_positions)
        best_gs = self.optimization_result['best_ground_station_position']
        degradations = [item['degradation'] for item in self.optimization_result['uav_degradations']]
        heights = [item['height'] for item in self.optimization_result['uav_degradations']]

        # 图1: 三维位置分布图
        ax1 = fig.add_subplot(131, projection='3d')

        # 绘制无人机位置
        scatter = ax1.scatter(uav_positions[:, 0], uav_positions[:, 1], uav_positions[:, 2],
                              c=degradations, cmap='RdYlBu_r', s=100, alpha=0.7)

        # 绘制地面站位置
        ax1.scatter(best_gs[0], best_gs[1], best_gs[2], c='red', marker='*',
                    s=200, label='最优地面站', edgecolors='black')

        # 绘制连接线
        for uav_pos in uav_positions:
            ax1.plot([uav_pos[0], best_gs[0]], [uav_pos[1], best_gs[1]], [uav_pos[2], best_gs[2]],
                     'gray', alpha=0.3, linestyle='--')

        ax1.set_xlabel('X坐标 (米)')
        ax1.set_ylabel('Y坐标 (米)')
        ax1.set_zlabel('高度 (米)')
        ax1.set_title('三维位置分布')
        ax1.legend()

        # 图2: 高度 vs 通信劣化程度
        ax2 = fig.add_subplot(132)
        scatter2 = ax2.scatter(heights, degradations, c=degradations,
                               cmap='RdYlBu_r', s=100, alpha=0.7)

        # 标记最优高度
        optimal_h = self.degradation_model.distance_params['h_optimal']
        ax2.axvline(x=optimal_h, color='green', linestyle='--', alpha=0.7, label=f'最优高度({optimal_h}米)')

        # 标记高度惩罚阈值
        penalty_h = self.degradation_model.distance_params['h_penalty_threshold']
        ax2.axvline(x=penalty_h, color='orange', linestyle='--', alpha=0.7, label=f'惩罚阈值({penalty_h}米)')

        ax2.set_xlabel('无人机高度 (米)')
        ax2.set_ylabel('通信劣化程度')
        ax2.set_title('高度 vs 通信劣化程度')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 图3: 距离 vs 通信劣化程度
        ax3 = fig.add_subplot(133)
        distances = [item['distance'] for item in self.optimization_result['uav_degradations']]
        scatter3 = ax3.scatter(distances, degradations, c=degradations,
                               cmap='RdYlBu_r', s=100, alpha=0.7)

        ax3.set_xlabel('三维距离 (米)')
        ax3.set_ylabel('通信劣化程度')
        ax3.set_title('距离 vs 通信劣化程度')
        ax3.grid(True, alpha=0.3)

        # 添加颜色条
        plt.colorbar(scatter, ax=ax1, label='劣化程度')
        plt.colorbar(scatter2, ax=ax2, label='劣化程度')
        plt.colorbar(scatter3, ax=ax3, label='劣化程度')

        plt.tight_layout()
        plt.show()


# 使用示例
def main():
    """
    主函数 - 演示如何使用地面站优化中间件
    """
    # 1. 生成模拟无人机位置 (在500x500区域内均匀分布，高度在50-400米之间)
    np.random.seed(42)  # 设置随机种子保证可重复性
    num_uavs = 8
    uav_positions = []

    # 在区域中心生成无人机群
    center_x, center_y = 500, 500
    radius = 200

    for i in range(num_uavs):
        angle = 2 * np.pi * i / num_uavs
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        # 高度在50-400米之间随机分布
        height = 50 + 350 * np.random.random()
        uav_positions.append((x, y, height))

    print("生成的无人机位置:")
    for i, pos in enumerate(uav_positions):
        print(f"无人机 {i + 1}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}米)")

    # 2. 创建优化器并运行优化
    optimizer = DistanceOptimizer(
        uav_positions=uav_positions,
        population_size=30,  # 较小的种群大小用于快速演示
        max_generations=50,  # 较少的代数用于快速演示
        gs_height_range=(0, 30)  # 地面站高度范围 0-30米
    )

    # 3. 运行优化
    result = optimizer.run_optimization()

    # 4. 显示详细结果
    print("\n=== 优化结果详情 ===")
    print(f"最优地面站位置: ({result['best_ground_station_position'][0]:.2f}, "
          f"{result['best_ground_station_position'][1]:.2f}, "
          f"{result['best_ground_station_position'][2]:.2f}米)")
    print(f"平均通信劣化程度: {result['best_fitness']:.4f}")

    print("\n各无人机通信劣化详情:")
    for i, uav_info in enumerate(result['uav_degradations']):
        print(f"无人机 {i + 1}: 距离={uav_info['distance']:.1f}m, "
              f"高度={uav_info['height']:.1f}m, "
              f"劣化程度={uav_info['degradation']:.4f}")

    # 5. 绘制结果图表
    optimizer.plot_optimization_results()

    return result


if __name__ == "__main__":
    # 运行演示
    result = main()