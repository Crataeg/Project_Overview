import numpy as np
import geatpy as ea
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class InterferenceDegradationModel:
    """
    干扰相关的通信劣化程度计算模型
    计算干扰源对无人机通信性能的影响，考虑距离和高度因素
    """

    def __init__(self):
        """初始化干扰劣化模型参数"""
        # 干扰相关参数
        self.interference_params = {
            'PLR0': 0.01,  # 基础丢包率
            'CR0': 0.02,  # 基础冲突率
            'S': 1.0,  # 信号功率 (归一化)
            'N': 0.1,  # 噪声功率 (归一化)
            'f_uav': 2400,  # 无人机工作频率 (MHz)
            'P_int_ref': 1.0,  # 参考干扰功率
            'BW_ref': 20,  # 参考干扰带宽 (MHz)
            'max_interference_distance': 2000,  # 最大干扰距离 (米)
            'h_max': 500,  # 最大允许高度 (米) - 与distance.py保持一致
            'h_optimal': 150,  # 最优通信高度 (米) - 与distance.py保持一致
            'h_penalty_threshold': 300  # 高度惩罚阈值 (米) - 与distance.py保持一致
        }

        # 权重系数 - 用户可调整 (γ系数)
        self.gamma_weights = {
            'I_loss': 0.14,  # 干扰损耗权重 γ₁
            'PLR': 0.14,  # 丢包率权重 γ₂
            'SNR_loss': 0.14,  # 信噪比损失权重 γ₃
            'P_outage': 0.10,  # 中断概率权重 γ₄
            'Delay': 0.10,  # 延迟权重 γ₅
            'Throughput_loss': 0.10,  # 吞吐量损失权重 γ₆
            'Handover_failure': 0.08,  # 切换失败权重 γ₇
            'Multi_hop_outage': 0.05,  # 多跳中断权重 γ₈
            'System_throughput_loss': 0.05,  # 系统吞吐量损失权重 γ₉
            'Collision_rate': 0.05,  # 冲突率权重 γ₁₀
            'Height_penalty': 0.05  # 高度惩罚权重 γ₁₁
        }

        # 系数参数 - 用户可调整 (k系数)
        self.coefficients = {
            'k10': 0.001,  # 干扰丢包率系数 k₁₀
            'k11': 0.0005,  # 干扰冲突率系数 k₁₁
            'k12': 0.002,  # 中断概率系数 k₁₂
            'k13': 0.0003,  # 延迟系数 k₁₃
            'k14': 0.0015,  # 吞吐量损失系数 k₁₄
            'k15': 0.0008,  # 切换失败系数 k₁₅
            'k16': 0.0004,  # 多跳中断系数 k₁₆
            'k17': 0.0002,  # 系统吞吐量损失系数 k₁₇
            'k18': 0.002  # 高度惩罚系数 k₁₈
        }

        # 预定义干扰频段
        self.interference_bands = {
            '4G': (2500, 2700),  # 4G频段 (MHz)
            '5G': (3400, 3600),  # 5G频段 (MHz)
            'WiFi': (2400, 2480),  # WiFi频段 (MHz)
            'Public_Safety': (400, 500),  # 公共安全频段 (MHz)
            'Military': (2000, 2200),  # 军用频段 (MHz)
            'Industrial': (900, 1000)  # 工业频段 (MHz)
        }

    def calculate_3d_distance(self, uav_pos: Tuple[float, float, float],
                              jammer_pos: Tuple[float, float, float]) -> float:
        """
        计算三维空间中的距离
        """
        dx = uav_pos[0] - jammer_pos[0]
        dy = uav_pos[1] - jammer_pos[1]
        dz = uav_pos[2] - jammer_pos[2]
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def height_penalty(self, uav_height: float) -> float:
        """
        计算高度惩罚因子 - 与distance.py保持一致
        在最优高度附近惩罚最小，随着偏离最优高度惩罚增加
        """
        h_optimal = self.interference_params['h_optimal']
        h_max = self.interference_params['h_max']
        h_penalty_threshold = self.interference_params['h_penalty_threshold']

        # 高度偏离最优值的绝对值
        height_deviation = abs(uav_height - h_optimal)

        if height_deviation <= 50:  # 在最优高度±50米内，惩罚很小
            penalty = 0.05 * (height_deviation / 50)
        elif height_deviation <= h_penalty_threshold:
            penalty = 0.05 + 0.3 * ((height_deviation - 50) / (h_penalty_threshold - 50))
        else:
            penalty = 0.35 + 0.6 * ((height_deviation - h_penalty_threshold) / (h_max - h_penalty_threshold))

        return min(penalty, 1.0)

    def spectrum_overlap_factor(self, f_j: float, bw_j: float) -> float:
        """
        计算频谱重叠因子 S(f_j, f_uav)
        表示干扰频段与无人机工作频段的重叠程度

        Args:
            f_j: 干扰中心频率 (MHz)
            bw_j: 干扰带宽 (MHz)

        Returns:
            频谱重叠因子 (0-1)
        """
        f_uav = self.interference_params['f_uav']

        # 计算频率差
        freq_diff = abs(f_j - f_uav)

        # 计算重叠程度 (使用高斯函数模拟)
        # 重叠程度随频率差增大而减小
        overlap = np.exp(-0.5 * (freq_diff / (bw_j / 2)) ** 2)

        return min(overlap, 1.0)

    def interference_loss(self, distance: float, f_j: float, bw_j: float, p_int: float,
                          uav_height: float = 150) -> float:
        """
        计算干扰损耗，考虑高度影响
        I_loss(f) = P_int(f_j) × BW(f_j) × S(f_j, f_uav) / distance_factor × height_factor

        Args:
            distance: 干扰源到无人机的三维距离 (米)
            f_j: 干扰中心频率 (MHz)
            bw_j: 干扰带宽 (MHz)
            p_int: 干扰功率
            uav_height: 无人机高度 (米)

        Returns:
            干扰损耗值
        """
        # 计算频谱重叠因子
        S = self.spectrum_overlap_factor(f_j, bw_j)

        # 距离衰减因子 (假设干扰功率随距离平方衰减)
        distance_factor = 1 + (distance / 1000) ** 2

        # 高度影响因子 (高度偏离最优值会增加干扰影响)
        height_factor = 1 + 0.5 * self.height_penalty(uav_height)

        # 计算总干扰损耗
        I_loss = (p_int * bw_j * S * height_factor) / distance_factor

        return I_loss

    def total_interference_power(self, distance: float, f_j: float, bw_j: float,
                                 p_int: float, uav_height: float = 150) -> float:
        """
        计算总干扰功率 (归一化)，考虑高度影响

        Args:
            distance: 干扰源到无人机的三维距离 (米)
            f_j: 干扰中心频率 (MHz)
            bw_j: 干扰带宽 (MHz)
            p_int: 干扰功率
            uav_height: 无人机高度 (米)

        Returns:
            归一化的总干扰功率 (0-1)
        """
        I_loss = self.interference_loss(distance, f_j, bw_j, p_int, uav_height)

        # 归一化到0-1范围
        max_I_loss = (self.interference_params['P_int_ref'] *
                      self.interference_params['BW_ref'] * 1.0 * 1.5)  # 最大频谱重叠为1，考虑高度因子1.5

        return min(I_loss / max_I_loss, 1.0)

    def packet_loss_rate(self, I_total: float, uav_height: float = 150) -> float:
        """
        计算干扰丢包率，考虑高度影响
        PLR(i) = PLR₀ + k₁₀·I_total + ΔPLR(h)

        Args:
            I_total: 总干扰功率 (归一化)
            uav_height: 无人机高度 (米)

        Returns:
            丢包率
        """
        base_plr = self.interference_params['PLR0'] + self.coefficients['k10'] * I_total

        # 高度对丢包率的影响
        height_impact = 0.02 * self.height_penalty(uav_height)

        return min(base_plr + height_impact, 1.0)

    def snr_loss(self, I_total: float, uav_height: float = 150) -> float:
        """
        计算干扰信噪比损失，考虑高度影响
        SNR_loss(i) = I_total/(S + N) + ΔSNR_loss(h)

        Args:
            I_total: 总干扰功率 (归一化)
            uav_height: 无人机高度 (米)

        Returns:
            信噪比损失
        """
        S = self.interference_params['S']
        N = self.interference_params['N']
        base_snr_loss = I_total / (S + N)

        # 高度对信噪比的影响
        height_impact = 0.1 * self.height_penalty(uav_height)

        return min(base_snr_loss + height_impact, 1.0)

    def outage_probability(self, I_total: float, uav_height: float = 150) -> float:
        """
        计算干扰中断概率，考虑高度影响
        P_outage(i) = 1 - exp(-k₁₂·I_total) + ΔP_outage(h)

        Args:
            I_total: 总干扰功率 (归一化)
            uav_height: 无人机高度 (米)

        Returns:
            中断概率
        """
        base_outage = 1 - np.exp(-self.coefficients['k12'] * I_total)

        # 高度对中断概率的影响
        height_impact = 0.1 * self.height_penalty(uav_height)

        return min(base_outage + height_impact, 1.0)

    def communication_delay(self, I_total: float, uav_height: float = 150) -> float:
        """
        计算通信延迟，考虑高度影响
        Delay(i) = k₁₃·I_total + ΔDelay(h)

        Args:
            I_total: 总干扰功率 (归一化)
            uav_height: 无人机高度 (米)

        Returns:
            通信延迟
        """
        base_delay = self.coefficients['k13'] * I_total

        # 高度对延迟的影响
        height_impact = 2 * self.height_penalty(uav_height)

        return base_delay + height_impact

    def throughput_loss(self, I_total: float, uav_height: float = 150) -> float:
        """
        计算吞吐量损失，考虑高度影响
        Throughput_loss(i) = k₁₄·I_total + ΔThroughput_loss(h)

        Args:
            I_total: 总干扰功率 (归一化)
            uav_height: 无人机高度 (米)

        Returns:
            吞吐量损失
        """
        base_throughput_loss = self.coefficients['k14'] * I_total

        # 高度对吞吐量的影响
        height_impact = 0.1 * self.height_penalty(uav_height)

        return min(base_throughput_loss + height_impact, 1.0)

    def handover_failure(self, I_total: float, uav_height: float = 150) -> float:
        """
        计算切换失败率，考虑高度影响
        Handover_failure(i) = k₁₅·I_total + ΔHandover_failure(h)

        Args:
            I_total: 总干扰功率 (归一化)
            uav_height: 无人机高度 (米)

        Returns:
            切换失败率
        """
        base_handover = self.coefficients['k15'] * I_total

        # 高度对切换失败率的影响
        height_impact = 0.05 * self.height_penalty(uav_height)

        return min(base_handover + height_impact, 1.0)

    def multi_hop_outage(self, I_total: float, uav_height: float = 150) -> float:
        """
        计算多跳中断概率，考虑高度影响
        Multi_hop_outage(i) = 1 - (1 - P_outage(i))^N
        简化假设：每跳干扰程度相同，N=3跳

        Args:
            I_total: 总干扰功率 (归一化)
            uav_height: 无人机高度 (米)

        Returns:
            多跳中断概率
        """
        single_hop_outage = self.outage_probability(I_total, uav_height)
        return 1 - (1 - single_hop_outage) ** 3

    def system_throughput_loss(self, I_total: float, uav_height: float = 150) -> float:
        """
        计算系统吞吐量损失，考虑高度影响
        System_throughput_loss(i) = Throughput_loss(i) × N_hop + ΔSystem_throughput_loss(h)
        假设N_hop=3

        Args:
            I_total: 总干扰功率 (归一化)
            uav_height: 无人机高度 (米)

        Returns:
            系统吞吐量损失
        """
        base_system_loss = self.throughput_loss(I_total, uav_height) * 3

        # 高度对系统吞吐量的影响
        height_impact = 0.1 * self.height_penalty(uav_height)

        return min(base_system_loss + height_impact, 1.0)

    def collision_rate(self, I_total: float, uav_height: float = 150) -> float:
        """
        计算干扰冲突率，考虑高度影响
        Collision_rate(i) = CR₀ + k₁₁·I_total + ΔCollision_rate(h)

        Args:
            I_total: 总干扰功率 (归一化)
            uav_height: 无人机高度 (米)

        Returns:
            冲突率
        """
        base_collision = self.interference_params['CR0'] + self.coefficients['k11'] * I_total

        # 高度对冲突率的影响
        height_impact = 0.03 * self.height_penalty(uav_height)

        return min(base_collision + height_impact, 1.0)

    def calculate_single_uav_degradation(self, distance: float, frequency: float,
                                         bandwidth: float, power: float,
                                         uav_height: float = 150) -> float:
        """
        计算单个无人机的干扰通信劣化程度，考虑高度影响
        D₃(i) = γ₁ × I_loss(f) + γ₂ × PLR(i) + ... + γ₁₁ × Height_penalty(h)

        Args:
            distance: 干扰源到无人机的三维距离 (米)
            frequency: 干扰中心频率 (MHz)
            bandwidth: 干扰带宽 (MHz)
            power: 干扰功率
            uav_height: 无人机高度 (米)

        Returns:
            干扰通信劣化程度 (0-1)
        """
        # 计算总干扰功率 (归一化)，考虑高度影响
        I_total = self.total_interference_power(distance, frequency, bandwidth, power, uav_height)

        # 归一化各分量到0-1范围
        components = {
            'I_loss': I_total,  # 干扰损耗直接使用归一化的总干扰功率
            'PLR': self.packet_loss_rate(I_total, uav_height),
            'SNR_loss': self.snr_loss(I_total, uav_height),
            'P_outage': self.outage_probability(I_total, uav_height),
            'Delay': min(self.communication_delay(I_total, uav_height) / 10.0, 1.0),  # 假设最大延迟10.0
            'Throughput_loss': self.throughput_loss(I_total, uav_height),
            'Handover_failure': self.handover_failure(I_total, uav_height),
            'Multi_hop_outage': self.multi_hop_outage(I_total, uav_height),
            'System_throughput_loss': self.system_throughput_loss(I_total, uav_height),
            'Collision_rate': self.collision_rate(I_total, uav_height),
            'Height_penalty': self.height_penalty(uav_height)
        }

        # 加权求和
        total_degradation = 0
        for component, weight in self.gamma_weights.items():
            total_degradation += weight * components[component]

        return min(total_degradation, 1.0)  # 确保在0-1范围内

    def get_interference_band_info(self, frequency: float) -> Tuple[str, float]:
        """
        获取干扰频段信息和频率匹配度

        Args:
            frequency: 干扰中心频率 (MHz)

        Returns:
            (频段名称, 频率匹配度)
        """
        best_match = "Unknown"
        best_match_quality = 0.0

        for band_name, (low_freq, high_freq) in self.interference_bands.items():
            # 计算频率匹配度 (在频段内为1，接近频段则匹配度降低)
            if low_freq <= frequency <= high_freq:
                match_quality = 1.0
            else:
                # 计算与最近频段边界的距离
                dist_to_low = abs(frequency - low_freq)
                dist_to_high = abs(frequency - high_freq)
                min_dist = min(dist_to_low, dist_to_high)
                # 匹配度随距离增大而减小
                match_quality = max(0, 1 - min_dist / 500)  # 500MHz范围内有影响

            if match_quality > best_match_quality:
                best_match = band_name
                best_match_quality = match_quality

        return best_match, best_match_quality


class InterferenceOptimizationProblem(ea.Problem):
    """
    干扰源参数优化问题定义
    遗传算法寻找使多无人机通信劣化程度最大的干扰源参数组合
    """

    def __init__(self, uav_positions: List[Tuple[float, float, float]],
                 degradation_model: InterferenceDegradationModel,
                 search_area: Tuple[float, float, float, float] = (0, 0, 1000, 1000),
                 frequency_range: Tuple[float, float] = (100, 6000),
                 bandwidth_range: Tuple[float, float] = (1, 100),
                 power_range: Tuple[float, float] = (0.1, 10.0),
                 jammer_height_range: Tuple[float, float] = (0, 50)):
        """
        初始化优化问题

        Args:
            uav_positions: 无人机位置列表 [(x1,y1,h1), (x2,y2,h2), ...]
            degradation_model: 通信劣化程度计算模型
            search_area: 干扰源位置搜索区域 (x_min, y_min, x_max, y_max)
            frequency_range: 干扰频率范围 (MHz)
            bandwidth_range: 干扰带宽范围 (MHz)
            power_range: 干扰功率范围
            jammer_height_range: 干扰源高度范围 (米)
        """
        self.uav_positions = np.array(uav_positions)
        self.degradation_model = degradation_model
        self.search_area = search_area
        self.frequency_range = frequency_range
        self.bandwidth_range = bandwidth_range
        self.power_range = power_range
        self.jammer_height_range = jammer_height_range

        # 问题定义
        name = 'Interference_Parameter_Optimization'  # 问题名称
        M = 1  # 目标维度 (单目标优化)
        maxormins = [-1]  # 目标最小最大化标记 (-1表示最大化，1表示最小化)
        Dim = 6  # 决策变量维度 (干扰源x,y,h坐标, 频率, 带宽, 功率)

        # 变量类型: 0-连续, 1-离散
        varTypes = [0, 0, 0, 0, 0, 0]  # 所有变量连续

        # 决策变量范围
        lb = [
            search_area[0], search_area[1], jammer_height_range[0],  # x,y,h坐标下界
            frequency_range[0],  # 频率下界
            bandwidth_range[0],  # 带宽下界
            power_range[0]  # 功率下界
        ]
        ub = [
            search_area[2], search_area[3], jammer_height_range[1],  # x,y,h坐标上界
            frequency_range[1],  # 频率上界
            bandwidth_range[1],  # 带宽上界
            power_range[1]  # 功率上界
        ]

        # 边界包含性 (1-包含边界，0-不包含)
        lbin = [1, 1, 1, 1, 1, 1]  # 包含所有下界
        ubin = [1, 1, 1, 1, 1, 1]  # 包含所有上界

        # 调用父类构造函数初始化问题
        super().__init__(name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        """
        目标函数 - 计算种群中每个个体的适应度

        Args:
            pop: 种群对象，包含决策变量矩阵 pop.Phen
        """
        # 获取决策变量矩阵，每一行代表一个干扰源参数组合 (x,y,h,频率,带宽,功率)
        x = pop.Phen  # 形状: (种群大小, 6)

        # 初始化适应度数组
        f_values = np.zeros((x.shape[0], 1))

        # 遍历种群中的每个个体
        for i in range(x.shape[0]):
            # 当前个体的干扰源参数
            interference_x = x[i, 0]  # 干扰源x坐标
            interference_y = x[i, 1]  # 干扰源y坐标
            interference_h = x[i, 2]  # 干扰源高度
            frequency = x[i, 3]  # 干扰频率 (MHz)
            bandwidth = x[i, 4]  # 干扰带宽 (MHz)
            power = x[i, 5]  # 干扰功率

            interference_pos = (interference_x, interference_y, interference_h)

            # 计算总通信劣化程度
            total_degradation = 0
            num_uavs = len(self.uav_positions)

            # 对每个无人机计算干扰劣化程度并求和
            for uav_pos in self.uav_positions:
                # 计算干扰源到无人机的三维距离
                distance = self.degradation_model.calculate_3d_distance(uav_pos, interference_pos)

                # 获取无人机高度
                uav_height = uav_pos[2]

                # 计算该干扰参数下的通信劣化程度
                uav_degradation = self.degradation_model.calculate_single_uav_degradation(
                    distance, frequency, bandwidth, power, uav_height)

                # 累加到总劣化程度
                total_degradation += uav_degradation

            # 计算平均劣化程度作为适应度值
            avg_degradation = total_degradation / num_uavs if num_uavs > 0 else 0

            # 存储适应度值
            f_values[i, 0] = avg_degradation

        # 设置种群适应度
        pop.ObjV = f_values


class InterferenceOptimizer:
    """
    干扰源参数优化器
    使用遗传算法寻找最优干扰源参数组合
    """

    def __init__(self,
                 uav_positions: List[Tuple[float, float, float]],
                 population_size: int = 50,
                 max_generations: int = 100,
                 jammer_height_range: Tuple[float, float] = (0, 50)):
        """
        初始化优化器

        Args:
            uav_positions: 无人机位置列表 (x, y, h)
            population_size: 种群大小
            max_generations: 最大进化代数
            jammer_height_range: 干扰源高度范围
        """
        self.uav_positions = uav_positions
        self.population_size = population_size
        self.max_generations = max_generations
        self.jammer_height_range = jammer_height_range

        # 创建通信劣化模型
        self.degradation_model = InterferenceDegradationModel()

        # 计算搜索区域 (基于无人机位置自动确定)
        uav_array = np.array(uav_positions)
        x_min, y_min = np.min(uav_array[:, :2], axis=0)
        x_max, y_max = np.max(uav_array[:, :2], axis=0)

        # 扩展搜索区域，给干扰源更多选择空间
        padding = max(x_max - x_min, y_max - y_min) * 0.5
        search_area = (x_min - padding, y_min - padding,
                       x_max + padding, y_max + padding)

        # 定义干扰参数范围
        frequency_range = (100, 6000)  # 频率范围 100-6000 MHz
        bandwidth_range = (1, 100)  # 带宽范围 1-100 MHz
        power_range = (0.1, 10.0)  # 功率范围 0.1-10.0

        # 创建优化问题
        self.problem = InterferenceOptimizationProblem(
            uav_positions, self.degradation_model, search_area,
            frequency_range, bandwidth_range, power_range, jammer_height_range)

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
        运行干扰源参数优化

        Returns:
            优化结果字典
        """
        print("开始干扰源参数优化...")
        print(f"无人机数量: {len(self.uav_positions)}")
        print(f"搜索区域: {self.problem.search_area}")
        print(f"干扰源高度范围: {self.jammer_height_range}米")
        print(f"频率范围: {self.problem.frequency_range} MHz")
        print(f"带宽范围: {self.problem.bandwidth_range} MHz")
        print(f"功率范围: {self.problem.power_range}")

        # 运行遗传算法
        [BestIndi, population] = self.algorithm.run()

        # 提取最优解
        best_solution = BestIndi.Phen[0]  # 最优干扰源参数 [x, y, h, 频率, 带宽, 功率]
        best_fitness = float(BestIndi.ObjV[0])  # 最优适应度值

        best_position = (best_solution[0], best_solution[1], best_solution[2])
        best_frequency = best_solution[3]
        best_bandwidth = best_solution[4]
        best_power = best_solution[5]

        # 获取频段信息
        band_name, match_quality = self.degradation_model.get_interference_band_info(best_frequency)

        # 计算最优干扰参数对各个无人机的劣化程度
        uav_degradations = []
        for i, uav_pos in enumerate(self.uav_positions):
            distance = self.degradation_model.calculate_3d_distance(uav_pos, best_position)
            uav_height = uav_pos[2]
            degradation = self.degradation_model.calculate_single_uav_degradation(
                distance, best_frequency, best_bandwidth, best_power, uav_height)

            uav_degradations.append({
                'uav_id': i + 1,
                'position': uav_pos,
                'distance_to_interference': distance,
                'height': uav_height,
                'degradation': degradation
            })

        # 计算总干扰功率
        avg_distance = np.mean([item['distance_to_interference'] for item in uav_degradations])
        avg_height = np.mean([item['height'] for item in uav_degradations])
        total_interference = self.degradation_model.total_interference_power(
            avg_distance, best_frequency, best_bandwidth, best_power, avg_height)

        # 构建结果字典
        self.optimization_result = {
            'best_interference_position': best_position,
            'best_frequency': best_frequency,
            'best_bandwidth': best_bandwidth,
            'best_power': best_power,
            'interference_band': band_name,
            'frequency_match_quality': match_quality,
            'total_interference_power': total_interference,
            'best_fitness': best_fitness,
            'uav_degradations': uav_degradations,
            'search_area': self.problem.search_area,
            'jammer_height_range': self.jammer_height_range,
            'parameter_ranges': {
                'frequency_range': self.problem.frequency_range,
                'bandwidth_range': self.problem.bandwidth_range,
                'power_range': self.problem.power_range
            },
            'optimization_parameters': {
                'population_size': self.population_size,
                'max_generations': self.max_generations
            }
        }

        print(f"优化完成!")
        print(f"最优干扰源位置: ({best_position[0]:.2f}, {best_position[1]:.2f}, {best_position[2]:.2f}米)")
        print(f"最优干扰频率: {best_frequency:.2f} MHz ({band_name}, 匹配度: {match_quality:.3f})")
        print(f"最优干扰带宽: {best_bandwidth:.2f} MHz")
        print(f"最优干扰功率: {best_power:.2f}")
        print(f"总干扰功率: {total_interference:.4f}")
        print(f"平均通信劣化程度: {best_fitness:.4f}")

        return self.optimization_result

    def plot_optimization_results(self):
        """
        绘制优化结果可视化图表
        """
        if self.optimization_result is None:
            print("请先运行优化!")
            return

        fig = plt.figure(figsize=(20, 12))

        # 提取数据
        uav_positions = np.array(self.uav_positions)
        best_interference = self.optimization_result['best_interference_position']
        degradations = [item['degradation'] for item in self.optimization_result['uav_degradations']]
        distances = [item['distance_to_interference'] for item in self.optimization_result['uav_degradations']]
        heights = [item['height'] for item in self.optimization_result['uav_degradations']]

        # 图1: 三维位置分布和干扰效应
        ax1 = fig.add_subplot(231, projection='3d')

        # 绘制无人机位置
        scatter1 = ax1.scatter(uav_positions[:, 0], uav_positions[:, 1], uav_positions[:, 2],
                               c=degradations, cmap='RdYlBu_r', s=100,
                               label='无人机位置', alpha=0.8, edgecolors='black')

        # 绘制最优干扰源位置
        ax1.scatter(best_interference[0], best_interference[1], best_interference[2],
                    c='red', marker='X', s=200, label='最优干扰源',
                    edgecolors='white', linewidth=2)

        # 绘制连接线
        for uav_pos in uav_positions:
            ax1.plot([uav_pos[0], best_interference[0]],
                     [uav_pos[1], best_interference[1]],
                     [uav_pos[2], best_interference[2]],
                     'gray', alpha=0.3, linestyle='--')

        ax1.set_xlabel('X坐标 (米)')
        ax1.set_ylabel('Y坐标 (米)')
        ax1.set_zlabel('高度 (米)')
        ax1.set_title('三维位置分布')
        ax1.legend()

        # 图2: 距离与通信劣化关系
        ax2 = fig.add_subplot(232)
        scatter2 = ax2.scatter(distances, degradations, c=degradations,
                               cmap='RdYlBu_r', s=100, alpha=0.7)
        ax2.set_xlabel('干扰源距离 (米)')
        ax2.set_ylabel('通信劣化程度')
        ax2.set_title('干扰距离 vs 通信劣化程度')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='劣化程度')

        # 图3: 高度与通信劣化关系
        ax3 = fig.add_subplot(233)
        scatter3 = ax3.scatter(heights, degradations, c=degradations,
                               cmap='RdYlBu_r', s=100, alpha=0.7)

        # 标记最优高度和惩罚阈值
        optimal_h = self.degradation_model.interference_params['h_optimal']
        penalty_h = self.degradation_model.interference_params['h_penalty_threshold']
        ax3.axvline(x=optimal_h, color='green', linestyle='--', alpha=0.7, label=f'最优高度({optimal_h}米)')
        ax3.axvline(x=penalty_h, color='orange', linestyle='--', alpha=0.7, label=f'惩罚阈值({penalty_h}米)')

        ax3.set_xlabel('无人机高度 (米)')
        ax3.set_ylabel('通信劣化程度')
        ax3.set_title('无人机高度 vs 通信劣化程度')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label='劣化程度')

        # 图4: 各无人机通信劣化程度
        ax4 = fig.add_subplot(234)
        uav_ids = [item['uav_id'] for item in self.optimization_result['uav_degradations']]
        bars = ax4.bar(uav_ids, degradations, color='lightcoral', alpha=0.7)

        # 为每个柱子添加数值标签
        for bar, degradation in zip(bars, degradations):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{degradation:.3f}', ha='center', va='bottom', fontsize=9)

        ax4.set_xlabel('无人机编号')
        ax4.set_ylabel('通信劣化程度')
        ax4.set_title('各无人机通信劣化程度')
        ax4.set_xticks(uav_ids)
        ax4.grid(True, alpha=0.3)

        # 图5: 干扰参数可视化
        ax5 = fig.add_subplot(235)
        result = self.optimization_result
        parameters = ['频率', '带宽', '功率', '干扰源高度']
        values = [result['best_frequency'], result['best_bandwidth'],
                  result['best_power'], result['best_interference_position'][2]]
        units = ['MHz', 'MHz', '', '米']
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']

        bars = ax5.bar(parameters, values, color=colors, alpha=0.7)

        # 为每个柱子添加数值标签
        for bar, value, unit in zip(bars, values, units):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height + max(values) * 0.05,
                     f'{value:.2f} {unit}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax5.set_ylabel('参数值')
        ax5.set_title('最优干扰参数')
        ax5.grid(True, alpha=0.3)

        # 图6: 高度惩罚函数可视化
        ax6 = fig.add_subplot(236)
        height_range = np.linspace(0, self.degradation_model.interference_params['h_max'], 100)
        penalties = [self.degradation_model.height_penalty(h) for h in height_range]

        ax6.plot(height_range, penalties, 'b-', linewidth=2)
        ax6.axvline(x=optimal_h, color='green', linestyle='--', alpha=0.7, label=f'最优高度({optimal_h}米)')
        ax6.axvline(x=penalty_h, color='orange', linestyle='--', alpha=0.7, label=f'惩罚阈值({penalty_h}米)')

        ax6.set_xlabel('无人机高度 (米)')
        ax6.set_ylabel('高度惩罚因子')
        ax6.set_title('高度惩罚函数')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_frequency_analysis(self):
        """
        绘制频率分析图表
        """
        if self.optimization_result is None:
            print("请先运行优化!")
            return

        result = self.optimization_result

        # 创建频率分析图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 图1: 频谱重叠因子随频率变化
        frequencies = np.linspace(100, 6000, 200)
        overlap_factors = []

        for freq in frequencies:
            overlap = self.degradation_model.spectrum_overlap_factor(
                freq, result['best_bandwidth'])
            overlap_factors.append(overlap)

        ax1.plot(frequencies, overlap_factors, 'b-', linewidth=2, label='频谱重叠因子')
        ax1.axvline(x=result['best_frequency'], color='red', linestyle='--',
                    label=f'最优频率: {result["best_frequency"]:.2f} MHz')
        ax1.axvline(x=self.degradation_model.interference_params['f_uav'],
                    color='green', linestyle='--',
                    label=f'无人机频率: {self.degradation_model.interference_params["f_uav"]} MHz')

        # 标记预定义频段
        for band_name, (low_freq, high_freq) in self.degradation_model.interference_bands.items():
            ax1.axvspan(low_freq, high_freq, alpha=0.2, label=band_name)

        ax1.set_xlabel('频率 (MHz)')
        ax1.set_ylabel('频谱重叠因子')
        ax1.set_title('频谱重叠因子 vs 干扰频率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 图2: 通信劣化程度随频率变化 (固定其他参数)
        avg_distance = np.mean([item['distance_to_interference'] for item in result['uav_degradations']])
        avg_height = np.mean([item['height'] for item in result['uav_degradations']])
        degradation_levels = []

        for freq in frequencies:
            degradation = self.degradation_model.calculate_single_uav_degradation(
                avg_distance, freq, result['best_bandwidth'], result['best_power'], avg_height)
            degradation_levels.append(degradation)

        ax2.plot(frequencies, degradation_levels, 'r-', linewidth=2, label='通信劣化程度')
        ax2.axvline(x=result['best_frequency'], color='red', linestyle='--',
                    label=f'最优频率: {result["best_frequency"]:.2f} MHz')

        ax2.set_xlabel('频率 (MHz)')
        ax2.set_ylabel('通信劣化程度')
        ax2.set_title('通信劣化程度 vs 干扰频率')
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
        summary = f"""
=== 干扰优化结果摘要 ===
最优干扰源位置: ({result['best_interference_position'][0]:.2f}, {result['best_interference_position'][1]:.2f}, {result['best_interference_position'][2]:.2f}米)
最优干扰频率: {result['best_frequency']:.2f} MHz
最优干扰带宽: {result['best_bandwidth']:.2f} MHz
最优干扰功率: {result['best_power']:.2f}
干扰频段: {result['interference_band']} (匹配度: {result['frequency_match_quality']:.3f})
总干扰功率: {result['total_interference_power']:.4f}
平均通信劣化程度: {result['best_fitness']:.4f}

各无人机详情:
"""
        for uav_info in result['uav_degradations']:
            summary += f"无人机 {uav_info['uav_id']}: 位置{uav_info['position']}, 高度{uav_info['height']:.1f}m, 距离干扰源{uav_info['distance_to_interference']:.1f}m, 劣化程度{uav_info['degradation']:.4f}\n"

        return summary


# 使用示例和测试函数
def main():
    """
    主函数 - 演示如何使用干扰优化中间件
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
    optimizer = InterferenceOptimizer(
        uav_positions=uav_positions,
        population_size=30,  # 较小的种群大小用于快速演示
        max_generations=50,  # 较少的代数用于快速演示
        jammer_height_range=(0, 30)  # 干扰源高度范围 0-30米
    )

    # 3. 运行优化
    result = optimizer.run_optimization()

    # 4. 显示详细结果
    print("\n" + "=" * 50)
    print(optimizer.get_optimization_summary())

    # 5. 绘制结果图表
    optimizer.plot_optimization_results()

    # 6. 绘制频率分析图表
    optimizer.plot_frequency_analysis()

    return result


if __name__ == "__main__":
    # 运行演示
    result = main()