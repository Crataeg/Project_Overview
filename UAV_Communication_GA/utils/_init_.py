"""
工具函数库
提供无人机通信优化系统的通用工具函数和可视化功能
"""

from .common_utils import (
    normalize_value,
    denormalize_value,
    calculate_distance,
    db_to_linear,
    linear_to_db,
    gaussian_function,
    sigmoid_function,
    calculate_euclidean_distance,
    calculate_manhattan_distance,
    calculate_angle_between_vectors,
    check_position_in_range,
    generate_random_positions,
    calculate_centroid,
    calculate_bounding_box,
    calculate_area,
    radians_to_degrees,
    degrees_to_radians,
    calculate_circle_area,
    calculate_sphere_volume,
    calculate_triangle_area,
    is_point_in_triangle,
    calculate_line_equation,
    calculate_perpendicular_distance,
    bound_value,
    linear_interpolation,
    logistic_function
)

from .visualization import (
    # 3D可视化函数
    plot_3d_optimization_results,
    plot_3d_comparison,
    plot_3d_position_distribution,

    # 分析可视化函数
    plot_degradation_components,
    plot_parameter_sensitivity,
    plot_convergence_history,

    # 报告生成函数
    create_optimization_report,
    save_plot,

    # 向后兼容的别名
    plot_optimization_results,
    plot_comparison,
    plot_position_distribution
)

__all__ = [
    # 数学工具函数
    'normalize_value',
    'denormalize_value',
    'bound_value',
    'linear_interpolation',

    # 单位转换函数
    'db_to_linear',
    'linear_to_db',
    'radians_to_degrees',
    'degrees_to_radians',

    # 数学函数
    'gaussian_function',
    'sigmoid_function',
    'logistic_function',

    # 几何计算函数
    'calculate_distance',
    'calculate_euclidean_distance',
    'calculate_manhattan_distance',
    'calculate_angle_between_vectors',
    'calculate_centroid',
    'calculate_bounding_box',
    'calculate_area',
    'calculate_circle_area',
    'calculate_sphere_volume',
    'calculate_triangle_area',
    'is_point_in_triangle',
    'calculate_line_equation',
    'calculate_perpendicular_distance',

    # 位置和范围函数
    'check_position_in_range',
    'generate_random_positions',

    # 3D可视化函数
    'plot_3d_optimization_results',
    'plot_3d_comparison',
    'plot_3d_position_distribution',

    # 分析可视化函数
    'plot_degradation_components',
    'plot_parameter_sensitivity',
    'plot_convergence_history',

    # 报告生成函数
    'create_optimization_report',
    'save_plot',

    # 向后兼容函数
    'plot_optimization_results',
    'plot_comparison',
    'plot_position_distribution'
]

__version__ = '2.0.0'
__author__ = 'UAV Communication Optimization System'

# 包级别的配置
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'color_palette': {
        'distance': '#1f77b4',
        'obstruction': '#ff7f0e',
        'interference': '#2ca02c',
        'velocity': '#d62728',
        'combined': '#9467bd'
    },
    'font_size': {
        'title': 14,
        'axis': 12,
        'legend': 10,
        'ticks': 10
    },
    '3d_plot': {
        'elevation': 30,
        'azimuth': 45,
        'marker_size': 100
    }
}


def get_module_info():
    """返回模块信息"""
    return {
        'name': 'UAV Utils Library',
        'version': __version__,
        'description': 'Provides general utility functions and 3D visualization capabilities for UAV communication optimization system',
        'author': __author__,
        'categories': {
            'math_utils': 'Mathematical utility functions',
            'conversion_utils': 'Unit conversion functions',
            'geometry_utils': 'Geometric calculation functions',
            'position_utils': 'Position and range functions',
            'visualization_3d': '3D visualization functions',
            'analysis_utils': 'Analysis and reporting functions'
        }
    }


def get_function_categories():
    """获取函数分类信息"""
    return {
        'mathematical_functions': [
            'normalize_value', 'denormalize_value', 'bound_value', 'linear_interpolation',
            'gaussian_function', 'sigmoid_function', 'logistic_function'
        ],
        'conversion_functions': [
            'db_to_linear', 'linear_to_db', 'radians_to_degrees', 'degrees_to_radians'
        ],
        'geometry_functions': [
            'calculate_distance', 'calculate_euclidean_distance', 'calculate_manhattan_distance',
            'calculate_angle_between_vectors', 'calculate_centroid', 'calculate_bounding_box',
            'calculate_area', 'calculate_circle_area', 'calculate_sphere_volume',
            'calculate_triangle_area', 'is_point_in_triangle', 'calculate_line_equation',
            'calculate_perpendicular_distance'
        ],
        'position_functions': [
            'check_position_in_range', 'generate_random_positions'
        ],
        'visualization_3d_functions': [
            'plot_3d_optimization_results', 'plot_3d_comparison', 'plot_3d_position_distribution'
        ],
        'analysis_functions': [
            'plot_degradation_components', 'plot_parameter_sensitivity', 'plot_convergence_history'
        ],
        'reporting_functions': [
            'create_optimization_report', 'save_plot'
        ],
        'compatibility_functions': [
            'plot_optimization_results', 'plot_comparison', 'plot_position_distribution'
        ]
    }


def test_all_utils():
    """测试所有工具函数"""
    print("Testing UAV Utils Library...")

    try:
        # 测试数学工具函数
        print("1. Testing mathematical functions:")
        norm_val = normalize_value(75, 0, 100)
        denorm_val = denormalize_value(0.75, 0, 100)
        print(f"  normalize_value(75, 0, 100) = {norm_val:.2f}")
        print(f"  denormalize_value(0.75, 0, 100) = {denorm_val:.2f}")

        # 测试单位转换
        print("2. Testing conversion functions:")
        linear_val = db_to_linear(20)  # 20 dB
        db_val = linear_to_db(100)  # 100倍
        print(f"  db_to_linear(20dB) = {linear_val:.2f}")
        print(f"  linear_to_db(100x) = {db_val:.2f} dB")

        # 测试几何计算
        print("3. Testing geometry functions:")
        dist = calculate_distance([0, 0], [3, 4])
        print(f"  calculate_distance([0,0], [3,4]) = {dist:.2f}")

        # 测试数学函数
        print("4. Testing mathematical functions:")
        gauss_val = gaussian_function(1.0, 0, 1)
        sigmoid_val = sigmoid_function(0, 0, 1)
        print(f"  gaussian_function(1.0, 0, 1) = {gauss_val:.4f}")
        print(f"  sigmoid_function(0, 0, 1) = {sigmoid_val:.4f}")

        # 测试3D位置生成
        print("5. Testing 3D position functions:")
        positions = generate_random_positions(3, (0, 100), (0, 100), (0, 50))
        centroid = calculate_centroid(positions)
        print(f"  Generated {len(positions)} random positions")
        print(f"  Centroid position: {centroid}")

        print("All utility function tests completed successfully!")
        return True

    except Exception as e:
        print(f"Utility function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_all_utils()