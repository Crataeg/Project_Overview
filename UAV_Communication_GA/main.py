#!/usr/bin/env python3
"""
无人机通信劣化程度优化系统 - 主程序入口（三维可视化版本）
总通信劣化程度 = W₁ × D₁(d) + W₂ × D₂(o) + W₃ × D₃(i) + W₄ × D₄(v)
"""

import numpy as np
import sys
import os
import argparse
from datetime import datetime
import traceback

# Keep the legacy console prints from crashing on GBK terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

print("=== 系统启动调试信息 ===")
print(f"当前目录: {current_dir}")
print(f"Python版本: {sys.version}")
print(f"工作目录: {os.getcwd()}")

# 调试信息：检查模块是否存在
print("\n=== 检查模块是否存在 ===")
modules_to_check = [
    ('distance', 'DistanceOptimizer'),
    ('obstruction', 'ObstructionOptimizer3D'),
    ('interference', 'InterferenceOptimizer'),
    ('velocity', 'VelocityOptimizer'),
    ('combined', 'CombinedOptimizer')
]

for package_name, class_name in modules_to_check:
    package_path = os.path.join(current_dir, package_name)
    module_path = os.path.join(package_path, f"{package_name}.py")
    if os.path.exists(package_path) and os.path.exists(module_path):
        print(f"✓ 找到 {package_name} 包和模块")
    else:
        print(f"✗ 未找到 {package_name} 包或模块")

# 导入各个模块 - 使用包结构导入
print("\n=== 尝试导入模块 ===")
try:
    from distance.distance import DistanceOptimizer

    print("✓ 成功导入 DistanceOptimizer")
except ImportError as e:
    print(f"✗ 导入 DistanceOptimizer 失败: {e}")
    traceback.print_exc()

try:
    from obstruction.obstruction import ObstructionOptimizer3D

    print("✓ 成功导入 ObstructionOptimizer3D")
except ImportError as e:
    print(f"✗ 导入 ObstructionOptimizer3D 失败: {e}")
    traceback.print_exc()

try:
    from interference.interference import InterferenceOptimizer

    print("✓ 成功导入 InterferenceOptimizer")
except ImportError as e:
    print(f"✗ 导入 InterferenceOptimizer 失败: {e}")
    traceback.print_exc()

try:
    from velocity.velocity import VelocityOptimizer

    print("✓ 成功导入 VelocityOptimizer")
except ImportError as e:
    print(f"✗ 导入 VelocityOptimizer 失败: {e}")
    traceback.print_exc()

try:
    from combined.combined import CombinedOptimizer

    print("✓ 成功导入 CombinedOptimizer")
except ImportError as e:
    print(f"✗ 导入 CombinedOptimizer 失败: {e}")
    traceback.print_exc()

# 导入其他工具模块
try:
    # 尝试导入三维可视化模块
    from utils.visualization import (
        plot_3d_optimization_results,
        plot_3d_comparison,
        plot_3d_position_distribution
    )
    from utils.common_utils import generate_random_positions, calculate_centroid

    print("✓ 成功导入三维可视化工具模块")

    # 设置别名以保持兼容性
    plot_optimization_results = plot_3d_optimization_results
    plot_comparison = plot_3d_comparison
    plot_position_distribution = plot_3d_position_distribution

except ImportError as e:
    print(f"✗ 导入三维可视化工具模块失败: {e}")
    try:
        # 尝试导入普通可视化模块
        from utils.visualization import (
            plot_optimization_results,
            plot_comparison,
            plot_position_distribution
        )
        from utils.common_utils import generate_random_positions, calculate_centroid

        print("✓ 成功导入二维可视化工具模块")
    except ImportError as e2:
        print(f"✗ 导入二维可视化工具模块也失败: {e2}")


        # 创建简单的替代函数
        def plot_optimization_results(*args, **kwargs):
            print("plot_optimization_results: 可视化功能不可用")
            # 尝试简单绘图
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, '优化结果可视化\n（三维可视化模块不可用）',
                         ha='center', va='center', fontsize=16)
                plt.title('优化结果')
                plt.axis('off')
                plt.show()
            except:
                pass


        def plot_comparison(*args, **kwargs):
            print("plot_comparison: 比较功能不可用")
            # 尝试简单绘图
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, '优化结果比较\n（三维可视化模块不可用）',
                         ha='center', va='center', fontsize=16)
                plt.title('优化结果比较')
                plt.axis('off')
                plt.show()
            except:
                pass


        def plot_position_distribution(*args, **kwargs):
            print("plot_position_distribution: 位置分布功能不可用")
            # 尝试简单绘图
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, '位置分布图\n（三维可视化模块不可用）',
                         ha='center', va='center', fontsize=16)
                plt.title('位置分布')
                plt.axis('off')
                plt.show()
            except:
                pass


        def generate_random_positions(num_positions, x_range, y_range, z_range):
            positions = []
            for i in range(num_positions):
                x = np.random.uniform(x_range[0], x_range[1])
                y = np.random.uniform(y_range[0], y_range[1])
                z = np.random.uniform(z_range[0], z_range[1])
                positions.append([x, y, z])
            return np.array(positions)


        def calculate_centroid(positions):
            return np.mean(positions, axis=0)

print("=== 调试信息结束 ===\n")


class UAVCommunicationOptimizationSystem:
    """
    无人机通信劣化程度优化系统主类（三维可视化版本）
    """

    def __init__(self, config_file: str = None):
        """
        初始化优化系统

        Args:
            config_file: 配置文件路径 (可选)
        """
        self.config = self._load_config(config_file)
        self.uav_positions = None
        self.results = {
            'distance': None,
            'obstruction': None,
            'interference': None,
            'velocity': None,
            'combined': None
        }
        self.optimization_history = []

        print("=" * 70)
        print("无人机通信劣化程度优化系统（三维可视化版本）")
        print("=" * 70)
        print("系统初始化完成!")

    def _load_config(self, config_file: str) -> dict:
        """
        加载配置文件

        Args:
            config_file: 配置文件路径

        Returns:
            配置字典
        """
        # 默认配置
        default_config = {
            'uav': {
                'num_uavs': 8,
                'area_range': (0, 1000),
                'height_range': (50, 100)
            },
            'optimization': {
                'population_size': 50,
                'max_generations': 100,
                'weights': {
                    'distance': 0.3,
                    'obstruction': 0.2,
                    'interference': 0.3,
                    'velocity': 0.2
                }
            },
            'output': {
                'save_results': True,
                'output_dir': 'results',
                'plot_format': ['png', 'pdf']
            }
        }

        # 如果提供了配置文件，可以在这里加载
        if config_file and os.path.exists(config_file):
            print(f"加载配置文件: {config_file}")
            # 这里可以添加配置文件解析逻辑
            # 例如: config = json.load(open(config_file, 'r'))

        return default_config

    def generate_uav_positions(self, num_uavs: int = None,
                               arrangement: str = 'circular') -> np.ndarray:
        """
        生成无人机位置

        Args:
            num_uavs: 无人机数量
            arrangement: 排列方式 ('circular', 'random', 'grid')

        Returns:
            无人机位置数组
        """
        if num_uavs is None:
            num_uavs = self.config['uav']['num_uavs']

        area_range = self.config['uav']['area_range']
        height_range = self.config['uav']['height_range']

        if arrangement == 'circular':
            # 圆形排列
            center_x = (area_range[0] + area_range[1]) / 2
            center_y = (area_range[0] + area_range[1]) / 2
            radius = min(area_range[1] - center_x, area_range[1] - center_y) * 0.8

            positions = []
            for i in range(num_uavs):
                angle = 2 * np.pi * i / num_uavs
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                z = np.random.uniform(height_range[0], height_range[1])
                positions.append([x, y, z])

            self.uav_positions = np.array(positions)

        elif arrangement == 'random':
            # 随机排列
            self.uav_positions = generate_random_positions(
                num_uavs, area_range, area_range, height_range)

        elif arrangement == 'grid':
            # 网格排列
            grid_size = int(np.ceil(np.sqrt(num_uavs)))
            positions = []
            spacing = (area_range[1] - area_range[0]) / (grid_size + 1)

            for i in range(num_uavs):
                row = i // grid_size
                col = i % grid_size
                x = area_range[0] + (col + 1) * spacing
                y = area_range[0] + (row + 1) * spacing
                z = np.random.uniform(height_range[0], height_range[1])
                positions.append([x, y, z])

            self.uav_positions = np.array(positions)

        print(f"生成 {num_uavs} 架无人机位置 ({arrangement}排列)")
        print(f"位置范围: X{area_range}, Y{area_range}, Z{height_range}")

        return self.uav_positions

    def run_distance_optimization(self, population_size: int = None,
                                  max_generations: int = None) -> dict:
        """
        运行距离优化

        Args:
            population_size: 种群大小
            max_generations: 最大代数

        Returns:
            优化结果
        """
        print("\n" + "=" * 50)
        print("开始距离优化...")
        print("=" * 50)

        if population_size is None:
            population_size = self.config['optimization']['population_size']
        if max_generations is None:
            max_generations = self.config['optimization']['max_generations']

        try:
            # 传递三维无人机位置 (x, y, h)
            optimizer = DistanceOptimizer(
                uav_positions=self.uav_positions.tolist(),  # 传递完整的三维位置
                population_size=population_size,
                max_generations=max_generations,
                gs_height_range=(0, 0)  # 地面站高度固定为0
            )

            result = optimizer.run_optimization()
            self.results['distance'] = result

            # 记录优化历史
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'type': 'distance',
                'result': result
            })
            print("✓ 距离优化完成")
            return result
        except Exception as e:
            print(f"❌ 距离优化失败: {e}")
            traceback.print_exc()
            return None

    def run_obstruction_optimization(self, population_size: int = None,
                                     max_generations: int = None,
                                     obstruction_strength: float = 1.0,
                                     obstruction_height: float = 50.0) -> dict:
        """
        运行遮挡优化

        Args:
            population_size: 种群大小
            max_generations: 最大代数
            obstruction_strength: 遮挡物强度
            obstruction_height: 遮挡物高度

        Returns:
            优化结果
        """
        print("\n" + "=" * 50)
        print("开始遮挡优化...")
        print("=" * 50)

        if population_size is None:
            population_size = self.config['optimization']['population_size']
        if max_generations is None:
            max_generations = self.config['optimization']['max_generations']

        try:
            # 传递三维无人机位置 (x, y, h)
            optimizer = ObstructionOptimizer3D(
                uav_positions=self.uav_positions.tolist(),  # 传递完整的三维位置
                population_size=population_size,
                max_generations=max_generations,
                obstruction_strength=obstruction_strength,
                obstruction_height=obstruction_height
            )

            result = optimizer.run_optimization()
            self.results['obstruction'] = result

            # 记录优化历史
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'type': 'obstruction',
                'result': result
            })
            print("✓ 遮挡优化完成")
            return result
        except Exception as e:
            print(f"❌ 遮挡优化失败: {e}")
            traceback.print_exc()
            return None

    def run_interference_optimization(self, population_size: int = None,
                                      max_generations: int = None) -> dict:
        """
        运行干扰优化

        Args:
            population_size: 种群大小
            max_generations: 最大代数

        Returns:
            优化结果
        """
        print("\n" + "=" * 50)
        print("开始干扰优化...")
        print("=" * 50)

        if population_size is None:
            population_size = self.config['optimization']['population_size']
        if max_generations is None:
            max_generations = self.config['optimization']['max_generations']

        try:
            # 传递三维无人机位置 (x, y, h)
            optimizer = InterferenceOptimizer(
                uav_positions=self.uav_positions.tolist(),  # 传递完整的三维位置
                population_size=population_size,
                max_generations=max_generations,
                jammer_height_range=(0, 50)  # 干扰源高度范围
            )

            result = optimizer.run_optimization()
            self.results['interference'] = result

            # 记录优化历史
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'type': 'interference',
                'result': result
            })
            print("✓ 干扰优化完成")
            return result
        except Exception as e:
            print(f"❌ 干扰优化失败: {e}")
            traceback.print_exc()
            return None

    def run_velocity_optimization(self, population_size: int = None,
                                  max_generations: int = None,
                                  velocity_range: tuple = (0, 50)) -> dict:
        """
        运行速度优化

        Args:
            population_size: 种群大小
            max_generations: 最大代数
            velocity_range: 速度范围

        Returns:
            优化结果
        """
        print("\n" + "=" * 50)
        print("开始速度优化...")
        print("=" * 50)

        if population_size is None:
            population_size = self.config['optimization']['population_size']
        if max_generations is None:
            max_generations = self.config['optimization']['max_generations']

        try:
            num_uavs = len(self.uav_positions)
            optimizer = VelocityOptimizer(
                num_uavs=num_uavs,
                population_size=population_size,
                max_generations=max_generations,
                velocity_range=velocity_range
            )

            result = optimizer.run_optimization()
            self.results['velocity'] = result

            # 记录优化历史
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'type': 'velocity',
                'result': result
            })
            print("✓ 速度优化完成")
            return result
        except Exception as e:
            print(f"❌ 速度优化失败: {e}")
            traceback.print_exc()
            return None

    def run_combined_optimization(self, population_size: int = None,
                                  max_generations: int = None,
                                  weights: dict = None) -> dict:
        """
        运行综合优化

        Args:
            population_size: 种群大小
            max_generations: 最大代数
            weights: 各因素权重

        Returns:
            优化结果
        """
        print("\n" + "=" * 50)
        print("开始综合优化...")
        print("=" * 50)

        if population_size is None:
            population_size = self.config['optimization']['population_size']
        if max_generations is None:
            max_generations = self.config['optimization']['max_generations']
        if weights is None:
            weights = self.config['optimization']['weights']

        try:
            optimizer = CombinedOptimizer(
                uav_positions=self.uav_positions.tolist(),  # 传递完整的三维位置
                population_size=population_size,
                max_generations=max_generations
            )

            # 设置权重
            optimizer.degradation_model.set_combined_weights(weights)

            result, fitness, history = optimizer.optimize()
            self.results['combined'] = {
                'solution': result,
                'fitness': fitness,
                'history': history
            }

            # 分析结果
            optimizer.analyze_results(result, fitness)

            # 记录优化历史
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'type': 'combined',
                'result': result
            })
            print("✓ 综合优化完成")
            return result
        except Exception as e:
            print(f"❌ 综合优化失败: {e}")
            traceback.print_exc()
            return None

    def run_all_optimizations(self, quick_mode: bool = False) -> dict:
        """
        运行所有优化

        Args:
            quick_mode: 快速模式 (使用较小的种群和代数)

        Returns:
            所有优化结果
        """
        print("\n" + "=" * 70)
        print("开始运行所有优化模块")
        print("=" * 70)

        if quick_mode:
            pop_size = 30
            max_gen = 50
            print("快速模式: 种群大小=30, 最大代数=50")
        else:
            pop_size = self.config['optimization']['population_size']
            max_gen = self.config['optimization']['max_generations']

        # 运行各个优化
        print("1. 运行距离优化...")
        self.run_distance_optimization(pop_size, max_gen)

        print("2. 运行遮挡优化...")
        self.run_obstruction_optimization(pop_size, max_gen)

        print("3. 运行干扰优化...")
        self.run_interference_optimization(pop_size, max_gen)

        print("4. 运行速度优化...")
        self.run_velocity_optimization(pop_size, max_gen)

        print("5. 运行综合优化...")
        self.run_combined_optimization(pop_size, max_gen)

        return self.results

    def compare_results(self):
        """比较各个优化结果"""
        print("\n" + "=" * 70)
        print("优化结果比较")
        print("=" * 70)

        # 提取各优化结果的最优适应度
        comparison_data = {}

        for opt_type, result in self.results.items():
            if result is not None:
                if opt_type == 'combined':
                    fitness = result['fitness']
                else:
                    fitness = result.get('best_fitness', 0)

                comparison_data[opt_type] = fitness

                print(f"{opt_type:>12} 优化: 劣化程度 = {fitness:.4f}")

        # 找出最优的优化方法
        if comparison_data:
            best_method = max(comparison_data, key=comparison_data.get)
            print(f"\n🎯 最优优化方法: {best_method}")
            print(f"📊 最大通信劣化程度: {comparison_data[best_method]:.4f}")

        return comparison_data

    def visualize_results(self, save_plots: bool = None):
        """
        可视化优化结果（三维版本）

        Args:
            save_plots: 是否保存图表
        """
        if save_plots is None:
            save_plots = self.config['output']['save_results']

        output_dir = self.config['output']['output_dir']

        print(f"\n生成三维可视化图表...")

        # 可视化综合优化结果
        if self.results['combined'] is not None:
            combined_result = self.results['combined']['solution']
            history = self.results['combined']['history']

            print("生成综合优化结果三维图...")
            plot_optimization_results(
                history, combined_result, self.uav_positions,
                save_path=output_dir if save_plots else None
            )

        # 比较各个优化结果
        comparison_data = self.compare_results()

        if len(comparison_data) > 1:
            # 修复：确保所有结果都存在
            individual_results = {}
            for opt_type in ['distance', 'obstruction', 'interference', 'velocity']:
                if self.results[opt_type] is not None:
                    individual_results[opt_type] = self.results[opt_type]
                else:
                    individual_results[opt_type] = {'fitness': 0}  # 默认值

            print("生成优化结果比较三维图...")
            plot_comparison(
                individual_results, self.results['combined'],
                save_path=output_dir if save_plots else None
            )

        # 三维位置分布图
        positions_data = {}
        if self.results['distance'] is not None:
            positions_data['gs_position'] = self.results['distance']['best_ground_station_position']
        if self.results['obstruction'] is not None:
            positions_data['obstruction_position'] = self.results['obstruction']['best_obstruction_position']
        if self.results['interference'] is not None:
            positions_data['interference_position'] = self.results['interference']['best_interference_position']

        if positions_data:
            print("生成位置分布三维图...")
            plot_position_distribution(
                positions_data, self.uav_positions,
                save_path=output_dir if save_plots else None
            )

    def generate_report(self):
        """生成优化报告"""
        print("\n" + "=" * 70)
        print("优化系统报告")
        print("=" * 70)

        print(f"无人机数量: {len(self.uav_positions)}")
        print(f"优化执行次数: {len(self.optimization_history)}")

        # 显示各个优化的简要结果
        for history in self.optimization_history:
            opt_type = history['type']
            timestamp = history['timestamp'].strftime("%Y-%m-%d %H:%M:%S")

            if opt_type == 'combined':
                fitness = history['result'].get('total_degradation', 'N/A')
            else:
                fitness = history['result'].get('best_fitness', 'N/A')

            print(f"{timestamp} - {opt_type:>12}: 劣化程度 = {fitness}")

        # 系统统计
        centroid = calculate_centroid(self.uav_positions)
        print(f"\n系统统计:")
        print(f"  无人机位置质心: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")
        print(f"  覆盖区域: {self.config['uav']['area_range']}")

        return self.optimization_history


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='无人机通信劣化程度优化系统（三维可视化版本）')
    parser.add_argument('--num-uavs', type=int, default=8, help='无人机数量')
    parser.add_argument('--arrangement', choices=['circular', 'random', 'grid'],
                        default='circular', help='无人机排列方式')
    parser.add_argument('--quick', action='store_true', help='快速模式')
    parser.add_argument('--save-plots', action='store_true', help='保存图表')
    parser.add_argument('--output-dir', default='results', help='输出目录')
    parser.add_argument('--obstruction-height', type=float, default=50.0,
                        help='遮挡物高度')
    parser.add_argument('--debug', action='store_true', help='调试模式')

    args = parser.parse_args()

    # 创建优化系统实例
    system = UAVCommunicationOptimizationSystem()

    # 更新配置
    system.config['uav']['num_uavs'] = args.num_uavs
    system.config['output']['output_dir'] = args.output_dir
    if args.save_plots:
        system.config['output']['save_results'] = True

    try:
        # 生成无人机位置
        print("\n步骤1: 生成无人机位置...")
        system.generate_uav_positions(args.num_uavs, args.arrangement)

        # 显示无人机位置
        print("\n无人机位置详情:")
        for i, pos in enumerate(system.uav_positions):
            print(f"  无人机 {i + 1}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

        # 运行所有优化
        print("\n步骤2: 运行所有优化...")
        system.run_all_optimizations(quick_mode=args.quick)

        # 可视化结果
        print("\n步骤3: 生成可视化结果...")
        system.visualize_results(save_plots=args.save_plots)

        # 生成报告
        print("\n步骤4: 生成报告...")
        system.generate_report()

        print("\n" + "=" * 70)
        print("优化系统运行完成!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n用户中断执行")
    except Exception as e:
        print(f"\n❌ 系统运行错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
