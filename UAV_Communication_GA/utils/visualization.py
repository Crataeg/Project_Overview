"""
三维可视化工具函数
提供无人机通信优化系统的三维数据可视化和结果展示功能
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Any, Optional
import os
import warnings

# 忽略字体相关的警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 完全使用英文，避免任何中文问题
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

# 颜色配置
COLOR_PALETTE = {
    'distance': '#1f77b4',  # 蓝色
    'obstruction': '#ff7f0e',  # 橙色
    'interference': '#2ca02c',  # 绿色
    'velocity': '#d62728',  # 红色
    'combined': '#9467bd',  # 紫色
    'background': '#f5f5f5',
    'grid': '#e0e0e0'
}


def plot_3d_optimization_results(history: Dict, best_solution: Dict,
                                 uav_positions: np.ndarray,
                                 save_path: Optional[str] = None) -> None:
    """
    Plot comprehensive 3D optimization results

    Args:
        history: Optimization history data
        best_solution: Best solution
        uav_positions: UAV positions (3D)
        save_path: Save path (optional)
    """
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('UAV Communication Degradation Optimization Results (3D)',
                 fontsize=16, fontweight='bold')

    # 1. 3D Position distribution plot
    ax1 = fig.add_subplot(231, projection='3d')
    _plot_3d_positions(ax1, best_solution, uav_positions)
    ax1.set_title('3D Position Distribution')

    # 2. Optimization convergence plot
    ax2 = fig.add_subplot(232)
    if 'best_fitness' in history and history['best_fitness']:
        ax2.plot(history['best_fitness'], 'b-', linewidth=2, label='Best Fitness')
        if 'mean_fitness' in history and history['mean_fitness']:
            ax2.plot(history['mean_fitness'], 'r--', linewidth=1, label='Average Fitness')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fitness Value')
        ax2.set_title('Optimization Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. Factor contribution pie chart
    ax3 = fig.add_subplot(233)
    factors = ['Distance', 'Obstruction', 'Interference', 'Velocity']
    if 'contributions' in best_solution:
        contributions = best_solution['contributions']
    else:
        contributions = [0.25, 0.25, 0.25, 0.25]

    colors = [COLOR_PALETTE['distance'], COLOR_PALETTE['obstruction'],
              COLOR_PALETTE['interference'], COLOR_PALETTE['velocity']]

    wedges, texts, autotexts = ax3.pie(contributions, labels=factors, colors=colors,
                                       autopct='%1.1f%%', startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax3.set_title('Factor Contribution Analysis')

    # 4. Parameter impact analysis
    ax4 = fig.add_subplot(234)
    param_names = ['Distance Degradation', 'Obstruction Degradation',
                   'Interference Degradation', 'Velocity Degradation']
    param_values = [
        best_solution.get('distance_degradation', 0),
        best_solution.get('obstruction_degradation', 0),
        best_solution.get('interference_degradation', 0),
        best_solution.get('velocity_degradation', 0)
    ]

    bars = ax4.bar(param_names, param_values, color=colors)
    ax4.set_ylabel('Degradation Level')
    ax4.set_title('Parameter Degradation Levels')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, value in zip(bars, param_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    # 5. UAV velocity distribution
    ax5 = fig.add_subplot(235)
    if 'uav_velocities' in best_solution:
        velocities = best_solution['uav_velocities']
        uav_ids = range(1, len(velocities) + 1)

        bars = ax5.bar(uav_ids, velocities, color=COLOR_PALETTE['velocity'], alpha=0.7)
        ax5.set_xlabel('UAV ID')
        ax5.set_ylabel('Speed (m/s)')
        ax5.set_title('UAV Optimal Speed Distribution')
        ax5.set_xticks(uav_ids)
        ax5.grid(True, alpha=0.3)

        # Add value labels
        for bar, velocity in zip(bars, velocities):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{velocity:.1f}', ha='center', va='bottom', fontsize=9)

    # 6. Interference parameter details
    ax6 = fig.add_subplot(236)
    if 'interference_params' in best_solution:
        int_params = best_solution['interference_params']
        param_names = ['Frequency', 'Bandwidth', 'Power']
        param_values = [
            int_params.get('frequency', 0) / 1e6,  # Convert to MHz
            int_params.get('bandwidth', 0) / 1e6,  # Convert to MHz
            int_params.get('power', 0)
        ]
        units = ['MHz', 'MHz', 'W']

        bars = ax6.bar(param_names, param_values,
                       color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.7)
        ax6.set_ylabel('Parameter Value')
        ax6.set_title('Interference Source Parameters')
        ax6.grid(True, alpha=0.3)

        # Add value labels
        for bar, value, unit in zip(bars, param_values, units):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., height + max(param_values) * 0.05,
                     f'{value:.1f} {unit}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path, '3d_optimization_results')
    else:
        plt.show()


def _plot_3d_positions(ax, best_solution, uav_positions):
    """Plot 3D position distribution"""
    # Plot UAV positions with height coloring
    scatter = ax.scatter(uav_positions[:, 0], uav_positions[:, 1], uav_positions[:, 2],
                         c=uav_positions[:, 2], cmap='viridis', s=100,
                         label='UAVs', alpha=0.8, depthshade=True)

    # Plot ground station position (at ground level)
    if 'gs_position' in best_solution:
        gs_pos = best_solution['gs_position']
        if len(gs_pos) == 2:  # 2D position, add z=0
            ax.scatter(gs_pos[0], gs_pos[1], 0,
                       c=COLOR_PALETTE['distance'], s=200, label='Ground Station',
                       marker='^', edgecolors='black')
        elif len(gs_pos) == 3:  # 3D position
            ax.scatter(gs_pos[0], gs_pos[1], gs_pos[2],
                       c=COLOR_PALETTE['distance'], s=200, label='Ground Station',
                       marker='^', edgecolors='black')

    # Plot obstruction position
    if 'obstruction_position' in best_solution:
        obs_pos = best_solution['obstruction_position']
        if len(obs_pos) == 2:  # 2D position, assume height=50
            ax.scatter(obs_pos[0], obs_pos[1], 50,
                       c=COLOR_PALETTE['obstruction'], s=200, label='Obstruction',
                       marker='s', edgecolors='black')
        elif len(obs_pos) == 3:  # 3D position
            ax.scatter(obs_pos[0], obs_pos[1], obs_pos[2],
                       c=COLOR_PALETTE['obstruction'], s=200, label='Obstruction',
                       marker='s', edgecolors='black')

    # Plot interference source position
    if 'interference_params' in best_solution:
        int_params = best_solution['interference_params']
        int_pos = int_params.get('position', [0, 0, 0])
        if len(int_pos) == 2:  # 2D position, assume height=30
            ax.scatter(int_pos[0], int_pos[1], 30,
                       c=COLOR_PALETTE['interference'], s=180, label='Interference Source',
                       marker='D', edgecolors='black')
        elif len(int_pos) == 3:  # 3D position
            ax.scatter(int_pos[0], int_pos[1], int_pos[2],
                       c=COLOR_PALETTE['interference'], s=180, label='Interference Source',
                       marker='D', edgecolors='black')

    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_zlabel('Z Coordinate (m)')
    ax.legend()

    # Add colorbar for height
    plt.colorbar(scatter, ax=ax, label='Height (m)')


def plot_3d_comparison(individual_results: Dict, combined_result: Dict,
                       save_path: Optional[str] = None) -> None:
    """
    Plot individual vs combined optimization comparison in 3D

    Args:
        individual_results: Individual optimization results
        combined_result: Combined optimization result
        save_path: Save path (optional)
    """
    fig = plt.figure(figsize=(18, 8))

    # 1. Performance comparison bar chart
    ax1 = fig.add_subplot(121)
    methods = ['Distance Opt', 'Obstruction Opt', 'Interference Opt', 'Velocity Opt', 'Combined Opt']
    performances = [
        individual_results.get('distance', {}).get('best_fitness', 0),
        individual_results.get('obstruction', {}).get('best_fitness', 0),
        individual_results.get('interference', {}).get('best_fitness', 0),
        individual_results.get('velocity', {}).get('best_fitness', 0),
        combined_result.get('fitness', 0) if combined_result else 0
    ]

    colors = [COLOR_PALETTE['distance'], COLOR_PALETTE['obstruction'],
              COLOR_PALETTE['interference'], COLOR_PALETTE['velocity'],
              COLOR_PALETTE['combined']]

    bars = ax1.bar(methods, performances, color=colors, alpha=0.8)
    ax1.set_ylabel('Communication Degradation')
    ax1.set_title('Individual vs Combined Optimization Performance')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, value in zip(bars, performances):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.4f}', ha='center', va='bottom')

    # 2. 3D Position comparison
    ax2 = fig.add_subplot(122, projection='3d')
    _plot_3d_comparison_positions(ax2, individual_results, combined_result)
    ax2.set_title('3D Position Comparison')

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path, '3d_optimization_comparison')
    else:
        plt.show()


def _plot_3d_comparison_positions(ax, individual_results, combined_result):
    """Plot 3D position comparison"""
    # Collect all positions for comparison
    positions = []
    labels = []
    colors = []
    markers = []

    # Distance optimization - ground station
    if 'distance' in individual_results and individual_results['distance']:
        dist_result = individual_results['distance']
        gs_pos = dist_result.get('best_ground_station_position', [0, 0])
        if len(gs_pos) == 2:
            positions.append([gs_pos[0], gs_pos[1], 0])
            labels.append('Distance Opt GS')
            colors.append(COLOR_PALETTE['distance'])
            markers.append('^')

    # Obstruction optimization - obstruction
    if 'obstruction' in individual_results and individual_results['obstruction']:
        obs_result = individual_results['obstruction']
        obs_pos = obs_result.get('best_obstruction_position', [0, 0, 0])
        if len(obs_pos) == 3:
            positions.append(obs_pos)
            labels.append('Obstruction Opt')
            colors.append(COLOR_PALETTE['obstruction'])
            markers.append('s')

    # Interference optimization - interference source
    if 'interference' in individual_results and individual_results['interference']:
        int_result = individual_results['interference']
        int_pos = int_result.get('best_interference_position', [0, 0, 0])
        if len(int_pos) == 3:
            positions.append(int_pos)
            labels.append('Interference Opt')
            colors.append(COLOR_PALETTE['interference'])
            markers.append('D')

    # Combined optimization positions
    if combined_result and 'solution' in combined_result:
        solution = combined_result['solution']

        # Ground station
        gs_pos = solution.get('gs_position', [0, 0])
        if len(gs_pos) == 2:
            positions.append([gs_pos[0], gs_pos[1], 0])
            labels.append('Combined GS')
            colors.append(COLOR_PALETTE['combined'])
            markers.append('^')

        # Obstruction
        obs_pos = solution.get('obstruction_position', [0, 0, 0])
        if len(obs_pos) == 3:
            positions.append(obs_pos)
            labels.append('Combined Obstruction')
            colors.append(COLOR_PALETTE['combined'])
            markers.append('s')

        # Interference
        int_params = solution.get('interference_params', {})
        int_pos = int_params.get('position', [0, 0, 0])
        if len(int_pos) == 3:
            positions.append(int_pos)
            labels.append('Combined Interference')
            colors.append(COLOR_PALETTE['combined'])
            markers.append('D')

    # Plot all positions
    for i, (pos, color, marker) in enumerate(zip(positions, colors, markers)):
        ax.scatter(pos[0], pos[1], pos[2],
                   c=color, marker=marker, s=150, label=labels[i], alpha=0.8)

    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_zlabel('Z Coordinate (m)')
    ax.legend()


def plot_3d_position_distribution(positions: Dict, uav_positions: np.ndarray,
                                  save_path: Optional[str] = None) -> None:
    """
    Plot 3D position distribution

    Args:
        positions: Various position data
        uav_positions: UAV positions (3D)
        save_path: Save path (optional)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot UAV positions with height coloring
    scatter = ax.scatter(uav_positions[:, 0], uav_positions[:, 1], uav_positions[:, 2],
                         c=uav_positions[:, 2], cmap='viridis', s=100,
                         label='UAVs', alpha=0.8, depthshade=True)

    # Plot other positions
    if 'gs_position' in positions:
        gs_pos = positions['gs_position']
        if len(gs_pos) == 2:
            ax.scatter(gs_pos[0], gs_pos[1], 0,
                       c=COLOR_PALETTE['distance'], s=200, label='Ground Station',
                       marker='^', edgecolors='black')
        elif len(gs_pos) == 3:
            ax.scatter(gs_pos[0], gs_pos[1], gs_pos[2],
                       c=COLOR_PALETTE['distance'], s=200, label='Ground Station',
                       marker='^', edgecolors='black')

    if 'obstruction_position' in positions:
        obs_pos = positions['obstruction_position']
        if len(obs_pos) == 2:
            ax.scatter(obs_pos[0], obs_pos[1], 50,
                       c=COLOR_PALETTE['obstruction'], s=200, label='Obstruction',
                       marker='s', edgecolors='black')
        elif len(obs_pos) == 3:
            ax.scatter(obs_pos[0], obs_pos[1], obs_pos[2],
                       c=COLOR_PALETTE['obstruction'], s=200, label='Obstruction',
                       marker='s', edgecolors='black')

    if 'interference_position' in positions:
        int_pos = positions['interference_position']
        if len(int_pos) == 2:
            ax.scatter(int_pos[0], int_pos[1], 30,
                       c=COLOR_PALETTE['interference'], s=180, label='Interference Source',
                       marker='D', edgecolors='black')
        elif len(int_pos) == 3:
            ax.scatter(int_pos[0], int_pos[1], int_pos[2],
                       c=COLOR_PALETTE['interference'], s=180, label='Interference Source',
                       marker='D', edgecolors='black')

    # Plot connection lines from ground station to UAVs
    if 'gs_position' in positions:
        gs_pos = positions['gs_position']
        gs_3d = [gs_pos[0], gs_pos[1], 0] if len(gs_pos) == 2 else gs_pos
        for uav_pos in uav_positions:
            ax.plot([uav_pos[0], gs_3d[0]], [uav_pos[1], gs_3d[1]], [uav_pos[2], gs_3d[2]],
                    'gray', alpha=0.3, linestyle='--')

    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_zlabel('Z Coordinate (m)')
    ax.set_title('3D System Position Distribution')
    ax.legend()

    # Add colorbar for UAV heights
    plt.colorbar(scatter, ax=ax, label='UAV Height (m)')

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path, '3d_position_distribution')
    else:
        plt.show()


def plot_degradation_components(degradation_data: Dict,
                                save_path: Optional[str] = None) -> None:
    """
    Plot degradation component analysis

    Args:
        degradation_data: Degradation data
        save_path: Save path (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Stacked degradation components
    factors = ['Distance', 'Obstruction', 'Interference', 'Velocity']
    colors = [COLOR_PALETTE['distance'], COLOR_PALETTE['obstruction'],
              COLOR_PALETTE['interference'], COLOR_PALETTE['velocity']]

    if 'uav_degradations' in degradation_data:
        uav_data = degradation_data['uav_degradations']
        uav_ids = [f'UAV{i + 1}' for i in range(len(uav_data))]

        # Extract degradation levels for each factor
        distance_degs = [data.get('distance_degradation', 0) for data in uav_data]
        obstruction_degs = [data.get('obstruction_degradation', 0) for data in uav_data]
        interference_degs = [data.get('interference_degradation', 0) for data in uav_data]
        velocity_degs = [data.get('velocity_degradation', 0) for data in uav_data]

        bottom = np.zeros(len(uav_ids))
        for i, (degs, color, label) in enumerate(zip(
                [distance_degs, obstruction_degs, interference_degs, velocity_degs],
                colors, factors
        )):
            ax1.bar(uav_ids, degs, bottom=bottom, color=color, label=label, alpha=0.8)
            bottom += np.array(degs)

        ax1.set_ylabel('Degradation Level')
        ax1.set_title('UAV Communication Degradation Component Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

    # 2. Factor correlation heatmap
    components = ['Distance', 'Obstruction', 'Interference', 'Velocity']
    correlation_matrix = np.identity(4)  # Default identity matrix

    # Here you can calculate actual correlation matrix
    # Using example data for now
    im = ax2.imshow(correlation_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)

    ax2.set_xticks(range(len(components)))
    ax2.set_yticks(range(len(components)))
    ax2.set_xticklabels(components)
    ax2.set_yticklabels(components)
    ax2.set_title('Factor Correlation Analysis')

    # Add value labels
    for i in range(len(components)):
        for j in range(len(components)):
            ax2.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                     ha='center', va='center', color='black' if abs(correlation_matrix[i, j]) < 0.5 else 'white')

    plt.colorbar(im, ax=ax2)
    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path, 'degradation_components')
    else:
        plt.show()


def plot_parameter_sensitivity(parameter_ranges: Dict, sensitivity_data: Dict,
                               save_path: Optional[str] = None) -> None:
    """
    Plot parameter sensitivity analysis

    Args:
        parameter_ranges: Parameter ranges
        sensitivity_data: Sensitivity data
        save_path: Save path (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    parameters = list(parameter_ranges.keys())[:4]  # Take first 4 parameters

    for i, param in enumerate(parameters):
        if i >= len(axes):
            break

        if param in sensitivity_data:
            values = sensitivity_data[param]['values']
            degradations = sensitivity_data[param]['degradations']

            axes[i].plot(values, degradations, 'b-', linewidth=2, marker='o')
            axes[i].set_xlabel(f'{param} Parameter Value')
            axes[i].set_ylabel('Communication Degradation')
            axes[i].set_title(f'{param} Parameter Sensitivity Analysis')
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path, 'parameter_sensitivity')
    else:
        plt.show()


def plot_convergence_history(history: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot convergence history

    Args:
        history: Optimization history data
        save_path: Save path (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if 'best_fitness' in history:
        ax.plot(history['best_fitness'], 'b-', linewidth=2, label='Best Fitness')

    if 'mean_fitness' in history:
        ax.plot(history['mean_fitness'], 'r--', linewidth=1, label='Average Fitness')

    if 'worst_fitness' in history:
        ax.plot(history['worst_fitness'], 'g:', linewidth=1, label='Worst Fitness')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness Value')
    ax.set_title('Optimization Convergence History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path, 'convergence_history')
    else:
        plt.show()


def create_optimization_report(optimization_results: Dict,
                               output_path: str = 'optimization_report.png') -> None:
    """
    Create comprehensive optimization report

    Args:
        optimization_results: Optimization results data
        output_path: Output file path
    """
    # Here you can generate more complex reports with multiple subplots and data tables
    # Currently creating a simple version
    fig = plt.figure(figsize=(16, 12))

    # Use GridSpec for complex layout
    gs = fig.add_gridspec(3, 3)

    # Add various charts...
    # Can be extended as needed

    plt.suptitle('UAV Communication Optimization System Comprehensive Report', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_plot(fig, os.path.dirname(output_path),
              os.path.basename(output_path).replace('.png', ''))


def save_plot(fig: plt.Figure, directory: str, filename: str,
              formats: List[str] = ['png', 'pdf']) -> None:
    """
    Save plot to file

    Args:
        fig: matplotlib figure object
        directory: Save directory
        filename: Filename (without extension)
        formats: Save format list
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    for fmt in formats:
        filepath = os.path.join(directory, f'{filename}.{fmt}')
        fig.savefig(filepath, dpi=300, bbox_inches='tight', format=fmt)
        print(f"Plot saved: {filepath}")


# 保持向后兼容性 - 为现有代码提供别名
plot_optimization_results = plot_3d_optimization_results
plot_comparison = plot_3d_comparison
plot_position_distribution = plot_3d_position_distribution


# Test function
def test_3d_visualization_functions():
    """Test 3D visualization functions"""
    print("Testing 3D visualization functions...")

    # Create test data
    test_history = {
        'best_fitness': [0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.88, 0.9, 0.91, 0.92],
        'mean_fitness': [0.05, 0.2, 0.4, 0.6, 0.7, 0.75, 0.78, 0.8, 0.82, 0.83]
    }

    test_solution = {
        'gs_position': [250, 250],
        'obstruction_position': [300, 300, 60],  # 3D position
        'interference_params': {
            'position': [200, 200, 40],  # 3D position
            'frequency': 2.4e9,
            'bandwidth': 20e6,
            'power': 2.0
        },
        'uav_velocities': [25, 30, 35, 28],
        'distance_degradation': 0.7,
        'obstruction_degradation': 0.6,
        'interference_degradation': 0.8,
        'velocity_degradation': 0.65,
        'contributions': [0.21, 0.12, 0.24, 0.13]
    }

    test_uav_positions = np.array([
        [100, 200, 50],
        [150, 180, 60],
        [200, 220, 55],
        [180, 150, 65]
    ])

    # Test 3D plotting functions
    try:
        plot_3d_optimization_results(test_history, test_solution, test_uav_positions)
        print("✓ 3D optimization results plot test passed")

        plot_3d_position_distribution(test_solution, test_uav_positions)
        print("✓ 3D position distribution plot test passed")

        # Test comparison with mock data
        individual_results = {
            'distance': {'best_fitness': 0.75, 'best_ground_station_position': [250, 250]},
            'obstruction': {'best_fitness': 0.68, 'best_obstruction_position': [280, 280, 55]},
            'interference': {'best_fitness': 0.82, 'best_interference_position': [190, 190, 35]},
            'velocity': {'best_fitness': 0.60}
        }
        combined_result = {'fitness': 0.85, 'solution': test_solution}

        plot_3d_comparison(individual_results, combined_result)
        print("✓ 3D comparison plot test passed")

        print("All 3D visualization function tests completed!")
        return True
    except Exception as e:
        print(f"✗ 3D visualization function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_3d_visualization_functions()