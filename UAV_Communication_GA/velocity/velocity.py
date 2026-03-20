import numpy as np
import geatpy as ea
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

class VelocityDegradationModel:
    """
    速度相关的通信劣化程度计算模型
    计算无人机速度对通信性能的影响
    """

    def __init__(self):
        """初始化速度劣化模型参数"""
        # 速度相关参数
        self.velocity_params = {
            'PLR0': 0.01,  # 基础丢包率
            'P_ho0': 0.05,  # 基础切换失败率
            'CR0': 0.02,  # 基础冲突率
            'f0': 2400,  # 载波频率 (MHz)
            'c': 3e8,  # 光速 (m/s)
            'v_max': 100,  # 最大速度 (m/s) - 无人机合理上限
            'v_switch_threshold': 50,  # 速度切换阈值 (m/s)
            'sampling_interval': 0.1  # 采样间隔 (s)
        }

        # 权重系数 - 用户可调整 (δ系数)
        self.delta_weights = {
            'Doppler': 0.15,  # 多普勒频移权重 δ₁
            'PLR': 0.15,  # 丢包率权重 δ₂
            'SNR_loss': 0.15,  # 信噪比损失权重 δ₃
            'P_outage': 0.10,  # 中断概率权重 δ₄
            'Delay': 0.10,  # 延迟权重 δ₅
            'Throughput_loss': 0.10,  # 吞吐量损失权重 δ₆
            'Handover_failure': 0.10,  # 切换失败权重 δ₇
            'Multi_hop_outage': 0.05,  # 多跳中断权重 δ₈
            'System_throughput_loss': 0.05,  # 系统吞吐量损失权重 δ₉
            'Collision_rate': 0.05  # 冲突率权重 δ₁₀
        }

        # 系数参数 - 用户可调整 (k系数)
        self.coefficients = {
            'k12': 0.001,  # 速度丢包率系数 k₁₂
            'k13': 0.002,  # 速度切换失败率系数 k₁₃
            'k14': 0.0005,  # 信噪比损失系数 k₁₄
            'k15': 0.0008,  # 中断概率系数 k₁₅
            'k16': 0.0003,  # 延迟系数 k₁₆
            'k17': 0.0012,  # 吞吐量损失系数 k₁₇
            'k18': 0.0004,  # 多跳中断系数 k₁₈
            'k19': 0.0002,  # 系统吞吐量损失系数 k₁₉
            'k20': 0.0006  # 冲突率系数 k₂₀
        }

    def doppler_shift(self, v: float) -> float:
        """
        计算多普勒频移
        Doppler(v) = (v·f₀)/c

        Args:
            v: 无人机速度 (m/s)

        Returns:
            多普勒频移 (Hz)
        """
        # 将载波频率从MHz转换为Hz
        f0_hz = self.velocity_params['f0'] * 1e6

        # 计算多普勒频移
        doppler = (v * f0_hz) / self.velocity_params['c']

        return doppler

    def packet_loss_rate(self, v: float) -> float:
        """
        计算速度丢包率
        PLR(v) = PLR₀ + k₁₂·v

        Args:
            v: 无人机速度 (m/s)

        Returns:
            丢包率
        """
        return self.velocity_params['PLR0'] + self.coefficients['k12'] * v

    def snr_loss(self, v: float) -> float:
        """
        计算速度信噪比损失
        SNR_loss(v) = k₁₄·v

        Args:
            v: 无人机速度 (m/s)

        Returns:
            信噪比损失
        """
        return self.coefficients['k14'] * v

    def outage_probability(self, v: float) -> float:
        """
        计算速度中断概率
        P_outage(v) = 1 - exp(-k₁₅·v)

        Args:
            v: 无人机速度 (m/s)

        Returns:
            中断概率
        """
        return 1 - np.exp(-self.coefficients['k15'] * v)

    def communication_delay(self, v: float) -> float:
        """
        计算通信延迟
        Delay(v) = k₁₆·v

        Args:
            v: 无人机速度 (m/s)

        Returns:
            通信延迟
        """
        return self.coefficients['k16'] * v

    def throughput_loss(self, v: float) -> float:
        """
        计算吞吐量损失
        Throughput_loss(v) = k₁₇·v

        Args:
            v: 无人机速度 (m/s)

        Returns:
            吞吐量损失
        """
        return self.coefficients['k17'] * v

    def handover_failure(self, v: float) -> float:
        """
        计算速度切换失败率
        Handover_failure(v) = P_ho₀ + k₁₃·v

        Args:
            v: 无人机速度 (m/s)

        Returns:
            切换失败率
        """
        return self.velocity_params['P_ho0'] + self.coefficients['k13'] * v

    def multi_hop_outage(self, v: float) -> float:
        """
        计算多跳中断概率
        Multi_hop_outage(v) = 1 - (1 - P_outage(v))^N
        简化假设：每跳速度影响相同，N=3跳

        Args:
            v: 无人机速度 (m/s)

        Returns:
            多跳中断概率
        """
        single_hop_outage = self.outage_probability(v)
        return 1 - (1 - single_hop_outage) ** 3

    def system_throughput_loss(self, v: float) -> float:
        """
        计算系统吞吐量损失
        System_throughput_loss(v) = Throughput_loss(v) × N_hop
        假设N_hop=3

        Args:
            v: 无人机速度 (m/s)

        Returns:
            系统吞吐量损失
        """
        return self.throughput_loss(v) * 3

    def collision_rate(self, v: float) -> float:
        """
        计算速度冲突率
        Collision_rate(v) = CR₀ + k₂₀·v

        Args:
            v: 无人机速度 (m/s)

        Returns:
            冲突率
        """
        return self.velocity_params['CR0'] + self.coefficients['k20'] * v

    def calculate_single_uav_degradation(self, v: float) -> float:
        """
        计算单个无人机的速度通信劣化程度
        D₄(v) = δ₁ × Doppler(v) + δ₂ × PLR(v) + ... + δ₁₀ × Collision_rate(v)

        Args:
            v: 无人机速度 (m/s)

        Returns:
            速度通信劣化程度 (0-1)
        """
        # 归一化各分量到0-1范围
        components = {
            'Doppler': min(self.doppler_shift(v) / 2000, 1.0),  # 假设最大多普勒频移2000Hz
            'PLR': min(self.packet_loss_rate(v), 1.0),
            'SNR_loss': min(self.snr_loss(v), 1.0),
            'P_outage': self.outage_probability(v),
            'Delay': min(self.communication_delay(v) / 5.0, 1.0),  # 假设最大延迟5.0
            'Throughput_loss': min(self.throughput_loss(v), 1.0),
            'Handover_failure': min(self.handover_failure(v), 1.0),
            'Multi_hop_outage': self.multi_hop_outage(v),
            'System_throughput_loss': min(self.system_throughput_loss(v), 1.0),
            'Collision_rate': min(self.collision_rate(v), 1.0)
        }

        # 加权求和
        total_degradation = 0
        for component, weight in self.delta_weights.items():
            total_degradation += weight * components[component]

        return min(total_degradation, 1.0)  # 确保在0-1范围内

    def get_velocity_category(self, v: float) -> str:
        """
        获取速度类别描述

        Args:
            v: 无人机速度 (m/s)

        Returns:
            速度类别描述
        """
        if v < 10:
            return "低速"
        elif v < 30:
            return "中速"
        elif v < 60:
            return "高速"
        else:
            return "超高速"


class VelocityOptimizationProblem(ea.Problem):
    """
    无人机速度优化问题定义
    遗传算法寻找使多无人机通信劣化程度最大的速度组合
    """

    def __init__(self,
                 num_uavs: int,
                 degradation_model: VelocityDegradationModel,
                 velocity_range: Tuple[float, float] = (0, 100)):
        """
        初始化优化问题

        Args:
            num_uavs: 无人机数量
            degradation_model: 通信劣化程度计算模型
            velocity_range: 速度范围 (最小速度, 最大速度)
        """
        self.num_uavs = num_uavs
        self.degradation_model = degradation_model
        self.velocity_range = velocity_range

        # 问题定义
        name = 'Velocity_Optimization'  # 问题名称
        M = 1  # 目标维度 (单目标优化)
        maxormins = [-1]  # 目标最小最大化标记 (-1表示最大化，1表示最小化)
        Dim = num_uavs  # 决策变量维度 (每个无人机的速度)

        # 变量类型: 0-连续, 1-离散
        varTypes = [0] * num_uavs  # 所有变量连续

        # 决策变量范围
        lb = [velocity_range[0]] * num_uavs  # 下界
        ub = [velocity_range[1]] * num_uavs  # 上界

        # 边界包含性 (1-包含边界，0-不包含)
        lbin = [1] * num_uavs  # 包含所有下界
        ubin = [1] * num_uavs  # 包含所有上界

        # 调用父类构造函数初始化问题
        super().__init__(name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        """
        目标函数 - 计算种群中每个个体的适应度

        Args:
            pop: 种群对象，包含决策变量矩阵 pop.Phen
        """
        # 获取决策变量矩阵，每一行代表一个速度组合 (v1, v2, ..., v_n)
        x = pop.Phen  # 形状: (种群大小, 无人机数量)

        # 初始化适应度数组
        f_values = np.zeros((x.shape[0], 1))

        # 遍历种群中的每个个体
        for i in range(x.shape[0]):
            # 当前个体的速度组合
            velocity_combination = x[i, :]  # [v1, v2, ..., v_n]

            # 计算总通信劣化程度
            total_degradation = 0

            # 对每个无人机计算速度劣化程度并求和
            for v in velocity_combination:
                # 计算该速度下的通信劣化程度
                uav_degradation = self.degradation_model.calculate_single_uav_degradation(v)

                # 累加到总劣化程度
                total_degradation += uav_degradation

            # 计算平均劣化程度作为适应度值
            avg_degradation = total_degradation / self.num_uavs

            # 存储适应度值
            f_values[i, 0] = avg_degradation

        # 设置种群适应度
        pop.ObjV = f_values


class VelocityOptimizer:
    """
    无人机速度优化器
    使用遗传算法寻找最优速度组合
    """

    def __init__(self,
                 num_uavs: int,
                 population_size: int = 50,
                 max_generations: int = 100,
                 velocity_range: Tuple[float, float] = (0, 100)):
        """
        初始化优化器

        Args:
            num_uavs: 无人机数量
            population_size: 种群大小
            max_generations: 最大进化代数
            velocity_range: 速度范围 (最小速度, 最大速度)
        """
        self.num_uavs = num_uavs
        self.population_size = population_size
        self.max_generations = max_generations
        self.velocity_range = velocity_range

        # 创建通信劣化模型
        self.degradation_model = VelocityDegradationModel()

        # 创建优化问题
        self.problem = VelocityOptimizationProblem(
            num_uavs, self.degradation_model, velocity_range)

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
        运行无人机速度优化

        Returns:
            优化结果字典
        """
        print("开始无人机速度优化...")
        print(f"无人机数量: {self.num_uavs}")
        print(f"速度范围: {self.velocity_range[0]} - {self.velocity_range[1]} m/s")

        # 运行遗传算法
        [BestIndi, population] = self.algorithm.run()

        # 提取最优解
        best_velocities = BestIndi.Phen[0]  # 最优速度组合 [v1, v2, ..., v_n]
        best_fitness = float(BestIndi.ObjV[0])  # 最优适应度值

        # 计算各无人机的具体劣化程度和速度类别
        uav_degradations = []
        for i, v in enumerate(best_velocities):
            degradation = self.degradation_model.calculate_single_uav_degradation(v)
            velocity_category = self.degradation_model.get_velocity_category(v)

            uav_degradations.append({
                'uav_id': i + 1,
                'velocity': v,
                'velocity_category': velocity_category,
                'degradation': degradation,
                'doppler_shift': self.degradation_model.doppler_shift(v)
            })

        # 计算统计信息
        avg_velocity = np.mean(best_velocities)
        max_velocity = np.max(best_velocities)
        min_velocity = np.min(best_velocities)

        # 构建结果字典
        self.optimization_result = {
            'best_velocities': best_velocities.tolist(),
            'best_fitness': best_fitness,
            'uav_degradations': uav_degradations,
            'statistics': {
                'average_velocity': avg_velocity,
                'max_velocity': max_velocity,
                'min_velocity': min_velocity,
                'velocity_range': self.velocity_range
            },
            'optimization_parameters': {
                'population_size': self.population_size,
                'max_generations': self.max_generations,
                'num_uavs': self.num_uavs
            }
        }

        print(f"优化完成!")
        print(f"平均速度: {avg_velocity:.2f} m/s")
        print(f"速度范围: {min_velocity:.2f} - {max_velocity:.2f} m/s")
        print(f"平均通信劣化程度: {best_fitness:.4f}")

        return self.optimization_result

    def plot_optimization_results(self):
        """
        绘制优化结果可视化图表
        """
        if self.optimization_result is None:
            print("请先运行优化!")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 提取数据
        result = self.optimization_result
        uav_ids = [item['uav_id'] for item in result['uav_degradations']]
        velocities = [item['velocity'] for item in result['uav_degradations']]
        degradations = [item['degradation'] for item in result['uav_degradations']]
        doppler_shifts = [item['doppler_shift'] for item in result['uav_degradations']]
        categories = [item['velocity_category'] for item in result['uav_degradations']]

        # 图1: 各无人机速度分布
        bars1 = ax1.bar(uav_ids, velocities, color='lightblue', alpha=0.7)

        # 为每个柱子添加数值标签
        for bar, velocity in zip(bars1, velocities):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{velocity:.1f}', ha='center', va='bottom', fontsize=9)

        ax1.set_xlabel('无人机编号')
        ax1.set_ylabel('速度 (m/s)')
        ax1.set_title('各无人机最优速度')
        ax1.set_xticks(uav_ids)
        ax1.grid(True, alpha=0.3)

        # 图2: 各无人机通信劣化程度
        bars2 = ax2.bar(uav_ids, degradations, color='lightcoral', alpha=0.7)

        # 为每个柱子添加数值标签
        for bar, degradation in zip(bars2, degradations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{degradation:.3f}', ha='center', va='bottom', fontsize=9)

        ax2.set_xlabel('无人机编号')
        ax2.set_ylabel('通信劣化程度')
        ax2.set_title('各无人机通信劣化程度')
        ax2.set_xticks(uav_ids)
        ax2.grid(True, alpha=0.3)

        # 图3: 速度与通信劣化关系
        scatter = ax3.scatter(velocities, degradations, c=degradations,
                              cmap='RdYlBu_r', s=100, alpha=0.7)

        # 添加趋势线
        z = np.polyfit(velocities, degradations, 2)
        p = np.poly1d(z)
        velocity_range = np.linspace(min(velocities), max(velocities), 100)
        ax3.plot(velocity_range, p(velocity_range), "r--", alpha=0.8, linewidth=2)

        ax3.set_xlabel('速度 (m/s)')
        ax3.set_ylabel('通信劣化程度')
        ax3.set_title('速度 vs 通信劣化程度')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='劣化程度')

        # 图4: 多普勒频移分析
        bars4 = ax4.bar(uav_ids, doppler_shifts, color='lightgreen', alpha=0.7)

        # 为每个柱子添加数值标签
        for bar, doppler in zip(bars4, doppler_shifts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 5,
                     f'{doppler:.1f}', ha='center', va='bottom', fontsize=9)

        ax4.set_xlabel('无人机编号')
        ax4.set_ylabel('多普勒频移 (Hz)')
        ax4.set_title('各无人机多普勒频移')
        ax4.set_xticks(uav_ids)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_velocity_degradation_analysis(self):
        """
        绘制速度与通信劣化关系分析图表
        """
        if self.optimization_result is None:
            print("请先运行优化!")
            return

        # 创建分析图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 图1: 速度与各通信指标关系
        velocities = np.linspace(0, self.velocity_range[1], 100)

        # 计算各指标随速度变化
        doppler_values = [self.degradation_model.doppler_shift(v) for v in velocities]
        plr_values = [self.degradation_model.packet_loss_rate(v) for v in velocities]
        snr_values = [self.degradation_model.snr_loss(v) for v in velocities]
        outage_values = [self.degradation_model.outage_probability(v) for v in velocities]
        handover_values = [self.degradation_model.handover_failure(v) for v in velocities]

        ax1.plot(velocities, doppler_values, 'b-', linewidth=2, label='多普勒频移 (Hz)')
        ax1.plot(velocities, plr_values, 'r-', linewidth=2, label='丢包率')
        ax1.plot(velocities, snr_values, 'g-', linewidth=2, label='信噪比损失')
        ax1.plot(velocities, outage_values, 'm-', linewidth=2, label='中断概率')
        ax1.plot(velocities, handover_values, 'c-', linewidth=2, label='切换失败率')

        # 标记最优速度范围
        result = self.optimization_result
        min_opt_velocity = min([item['velocity'] for item in result['uav_degradations']])
        max_opt_velocity = max([item['velocity'] for item in result['uav_degradations']])

        ax1.axvspan(min_opt_velocity, max_opt_velocity, alpha=0.2, color='orange',
                    label='最优速度范围')

        ax1.set_xlabel('速度 (m/s)')
        ax1.set_ylabel('指标值')
        ax1.set_title('速度与各通信指标关系')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 图2: 速度与总通信劣化程度关系
        degradation_values = [self.degradation_model.calculate_single_uav_degradation(v)
                              for v in velocities]

        ax2.plot(velocities, degradation_values, 'k-', linewidth=3, label='总通信劣化程度')

        # 标记最优速度点
        for uav_info in result['uav_degradations']:
            ax2.axvline(x=uav_info['velocity'], color='red', linestyle='--', alpha=0.5)

        ax2.axvspan(min_opt_velocity, max_opt_velocity, alpha=0.2, color='orange',
                    label='最优速度范围')

        ax2.set_xlabel('速度 (m/s)')
        ax2.set_ylabel('总通信劣化程度')
        ax2.set_title('速度 vs 总通信劣化程度')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

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
        stats = result['statistics']

        summary = f"""
=== 速度优化结果摘要 ===
无人机数量: {self.num_uavs}
平均速度: {stats['average_velocity']:.2f} m/s
速度范围: {stats['min_velocity']:.2f} - {stats['max_velocity']:.2f} m/s
平均通信劣化程度: {result['best_fitness']:.4f}

各无人机详情:
"""
        for uav_info in result['uav_degradations']:
            summary += f"无人机 {uav_info['uav_id']}: 速度{uav_info['velocity']:.2f}m/s ({uav_info['velocity_category']}), 多普勒频移{uav_info['doppler_shift']:.1f}Hz, 劣化程度{uav_info['degradation']:.4f}\n"

        # 添加速度类别统计
        categories = [item['velocity_category'] for item in result['uav_degradations']]
        category_counts = {}
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1

        summary += "\n速度类别统计:\n"
        for category, count in category_counts.items():
            percentage = (count / self.num_uavs) * 100
            summary += f"  {category}: {count}架 ({percentage:.1f}%)\n"

        return summary


# 使用示例和测试函数
def main():
    """
    主函数 - 演示如何使用速度优化中间件
    """
    # 1. 设置无人机数量和参数
    num_uavs = 8
    print(f"无人机数量: {num_uavs}")

    # 2. 创建优化器并运行优化
    optimizer = VelocityOptimizer(
        num_uavs=num_uavs,
        population_size=30,  # 较小的种群大小用于快速演示
        max_generations=50,  # 较少的代数用于快速演示
        velocity_range=(0, 80)  # 速度范围 0-80 m/s (无人机合理上限)
    )

    # 3. 运行优化
    result = optimizer.run_optimization()

    # 4. 显示详细结果
    print("\n" + "=" * 50)
    print(optimizer.get_optimization_summary())

    # 5. 绘制结果图表
    optimizer.plot_optimization_results()

    # 6. 绘制速度分析图表
    optimizer.plot_velocity_degradation_analysis()

    return result


if __name__ == "__main__":
    # 运行演示
    result = main()