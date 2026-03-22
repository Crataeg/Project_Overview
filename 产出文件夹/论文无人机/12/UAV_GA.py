"""
无人机通信性能劣化场景优化 - 高级遗传算法实现
将无人机高度、干扰源参数作为染色体进行优化
干扰源与建筑物紧密结合，考虑实际城市环境
支持在建筑物和无建筑物位置生成干扰源
"""

import numpy as np
import geatpy as ea
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import os
import json
import sys
from typing import List, Dict, Tuple, Optional

# 创建输出目录
os.makedirs("output", exist_ok=True)


# 设置中文字体支持
def set_chinese_font():
    """设置中文字体支持"""
    try:
        # 尝试不同平台的中文字体
        if matplotlib.get_backend().lower() in ['agg', 'tkagg', 'gtkagg', 'wxagg']:
            # 对于非交互式后端，需要特殊处理
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        else:
            # 对于交互式显示
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

        # 设置字体大小
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11

    except Exception as e:
        print(f"字体设置警告: {e}")
        print("继续使用默认字体...")


# 初始化中文字体
set_chinese_font()


class InterferenceSourceConfig:
    """干扰源配置类 - 定义不同干扰源的特性"""

    def __init__(self):
        # 定义干扰源类型及其特性
        self.interference_configs = {
            'wifi_2_4g': {
                'name': 'WiFi 2.4GHz AP',
                'frequency': 2.4415e9,
                'frequency_range': (2.400e9, 2.483e9),
                'evidence_screenshots': {'frequency': 'fcc_15_247_24ghz_band.png', 'background_PPP': 'stochastic_ppp_3d.png'},
                'power_range': (5, 25),  # dBm（下界锚点：2.4GHz 最低 5 dBm）
                'typical_height_range': (5, 50),  # 米，通常部署在室内或低矮建筑
                'height_distribution': 'building_low',  # 建筑物低层
                'path_loss_factor': 3.0,  # 路径损耗系数
                'coverage_radius': 100,  # 覆盖半径（米）
                'description': 'WiFi接入点，常见于室内和公共场所',
                'color': '#FF6B6B',  # 红色
                'calibration': {
                    'eirp_min_dbm': 5.0,  # 锚点：Wi‑Fi 2.4GHz 最低发射功率门槛
                    'wifi_beacon_period_s': 0.1024,  # 锚点：beacon 周期 102.4 ms
                    'wifi_beacon_duration_ms_default': 0.5,  # 经验默认（可由 user_params 覆盖）
                    'wifi_traffic_duty_cycle_default': 0.15,  # 经验默认（可由 user_params 覆盖）
                    'sources': {
                        'eirp_min_dbm': '图片大纲：Wi‑Fi 2.4GHz 最低发射功率门槛 5 dBm',
                        'wifi_beacon_period_s': '图片大纲：Wi‑Fi beacon 周期 102.4 ms'
                    }
                }
            },
            'wifi_5_8g': {
                'name': 'WiFi 5.8GHz AP',
                'frequency': 5.7875e9,
                'frequency_range': (5.725e9, 5.850e9),
                'evidence_screenshots': {'frequency': 'fcc_15_247_58ghz_band.png', 'background_PPP': 'stochastic_ppp_3d.png'},
                'power_range': (8, 25),  # dBm（下界锚点：5GHz 最低 8 dBm）
                'typical_height_range': (5, 50),  # 米
                'height_distribution': 'building_low',
                'path_loss_factor': 3.5,
                'coverage_radius': 80,
                'description': 'WiFi 5G接入点，穿墙能力稍弱',
                'color': '#4ECDC4',  # 青色
                'calibration': {
                    'eirp_min_dbm': 8.0,  # 锚点：Wi‑Fi 5GHz 最低发射功率门槛
                    'wifi_beacon_period_s': 0.1024,
                    'wifi_beacon_duration_ms_default': 0.5,
                    'wifi_traffic_duty_cycle_default': 0.15,
                    'sources': {
                        'eirp_min_dbm': '图片大纲：Wi‑Fi 5GHz 最低发射功率门槛 8 dBm',
                        'wifi_beacon_period_s': '图片大纲：Wi‑Fi beacon 周期 102.4 ms'
                    }
                }
            },
            'cellular_4g': {
                'name': '4G基站',
                'frequency': 2.6e9,
                'power_range': (40, 55),  # dBm
                'typical_height_range': (20, 100),  # 米，通常在建筑物顶部
                'height_distribution': 'building_top',
                'path_loss_factor': 2.7,
                'coverage_radius': 500,
                'description': '4G移动通信基站',
                'color': '#45B7D1'  # 蓝色
            },
            'cellular_5g': {
                'name': '5G基站',
                'frequency': 3.6e9,
                'frequency_range': (3.4e9, 3.8e9),
                'evidence_screenshots': {'frequency': 'dominance_n78_midband.png', 'power_range': 'tr138913_p27_power_46dBm_23d*.png'},
                # 图片大纲建议将“5G小基站”按 femto/pico/micro 分档；这里将 5G 默认按“小基站档位”标定
                'power_range': (20, 46),  # dBm（覆盖 femto/pico/micro + 适度余量）
                'typical_height_range': (20, 80),  # 米，部署更密集
                'height_distribution': 'building_top',
                'path_loss_factor': 2.8,
                'coverage_radius': 300,
                'description': '5G移动通信基站',
                'color': '#96CEB4',  # 绿色
                'calibration': {
                    'eirp_anchors_dbm': {
                        'femto': 20.0,
                        'pico': 24.0,
                        'micro': 36.0
                    },
                    'activity_factor': 0.70,  # 锚点：用 70% 发射活动因子把峰值 EIRP 折算成平均干扰强度
                    'sources': {
                        'eirp_anchors_dbm': '图片大纲：5G小基站 femto/pico/micro = 20/24/36 dBm',
                        'activity_factor': '图片大纲：发射活动因子 70%'
                    }
                }
            },
            'cellular_ue_ul': {
                'name': '蜂窝UE上行',
                'frequency': 2.6e9,
                'power_range': (0, 23),  # dBm（典型 UE Pmax=23 dBm）
                'typical_height_range': (0, 300),  # 米（地面 UE ~0；空中 UE 可到数百米）
                'height_distribution': 'building_low',
                'path_loss_factor': 2.5,
                'coverage_radius': 300,
                'description': '蜂窝 UE 上行（分数功控 FPC 标定）',
                'color': '#A29BFE',
                'calibration': {
                    'ul_fpc': {
                        'pmax_dbm': 23.0,
                        'p0_candidates_dbm': [-83.0, -88.0, -95.0, -101.0, -110.0],
                        'alpha_ground_or_low_uav': 0.8,
                        'alpha_high_uav': 0.7,
                        'alpha_switch_height_m': 100.0,
                        'm_rb_candidates': [1, 5, 10, 25, 50],
                        'delta_tf_db': 0.0,
                        'f_delta_tpc_db': 0.0,
                        'pl_extra_db_default': 20.0,
                        'serving_distance_m_range': (30.0, 500.0),
                    },
                    'sources': {
                        'ul_fpc': '图片大纲：分数功控公式 + α(高度分段) + TR 36.777 典型 (α,P0) 组合'
                    }
                }
            },
            'gnss_jammer': {
                'name': 'GNSS干扰机',
                'frequency': 1.575e9,
                'evidence_screenshots': {'frequency': 'isgps_l1_1575_42.png', 'power_range': 'jammertest23_power_settings.png'},
                'power_range': (20, 43),  # dBm
                'typical_height_range': (5, 30),  # 米，地面或低矮建筑
                'height_distribution': 'ground',
                'path_loss_factor': 2.2,
                'coverage_radius': 200,
                'description': 'GPS/北斗导航干扰设备',
                'color': '#FFEAA7',  # 黄色
                # 口径钉死：本工程 power_range(dBm) 在链路预算里作为 P_tx[dBm] 使用，等效为 EIRP-equivalent（天线增益/馈损等折算进该项）
                'power_semantics': 'eirp_equivalent_dbm',
                'calibration': {
                    'eirp_range_dbm': (20.0, 40.0),
                    'source_type': 'portable/vehicular GNSS jammer (EIRP-equivalent)',
                    'notes': [
                        '本工程使用 EIRP-equivalent dBm 口径（P_rx=P_tx-PL+...），因此该区间应理解为等效辐射功率范围',
                        '上限 40 dBm 可作为“可实现输出上限以下的保守值”，下限 20 dBm 作为场景轻量级覆盖'
                    ],
                    'sources': {
                        'eirp_range_dbm': '待补齐：ESA Jammertest 2023（文档提到最高约 43 dBm / 20 W 输出量级；本代码取 20–40 dBm 作为保守扫描范围）'
                    }
                }
            },
            'industrial_device': {
                'name': '工业设备干扰',
                'frequency': 915e6,
                'frequency_range': (902e6, 928e6),
                'evidence_screenshots': {'frequency': 'fcc_15_247_902_928_band.png'},
                'power_range': (25, 45),  # dBm
                'typical_height_range': (5, 100),
                'height_distribution': 'building_mid',
                'path_loss_factor': 2.5,
                'coverage_radius': 150,
                'description': '工业设备产生的电磁干扰',
                'color': '#DDA0DD',  # 紫色
                'power_semantics': 'eirp_equivalent_dbm',
                'calibration': {
                    'eirp_range_dbm': (25.0, 45.0),
                    'band_hz': 900e6,
                    'assumption': 'active transmitter in 902–928 MHz ISM band (EIRP-equivalent), not broadband EMI/PSD',
                    'eirp_mapping': 'P_EIRP[dBm] = P_cond[dBm] + G_t[dBi] - L_t[dB]',
                    'anchors': {
                        'p_cond_max_dbm': 30.0,
                        'eirp_upper_example_dbm': 45.0,
                        'eirp_upper_example_condition': '示例：P_cond=30 dBm + G_t≈15 dBi（定向天线）- L_t≈0 dB => P_EIRP≈45 dBm'
                    },
                    'notes': [
                        '若你要表达“工业EMI噪声源”，应改用 PSD/场强口径（dBm/Hz 或 dBµV/m），而不是 power_range(dBm)',
                        '当前实现是“单频/窄带等效发射机”模型：frequency+P_tx 直接进入 P_rx=P_tx-PL+...'
                    ],
                    'sources': {
                        'eirp_range_dbm': '待补齐：FCC 47 CFR §15.247（902–928 MHz ISM 扩频/数字调制设备的 conducted power 上限与天线增益相关约束；本代码将其折算为 EIRP-equivalent 上界示例）'
                    }
                }
            },
            'satellite_ground': {
                'name': '卫星地面站',
                'frequency': 14.25e9,
                'frequency_range': (14.0e9, 14.5e9),
                'evidence_screenshots': {'frequency': 'ku_buc_14ghz_power.png'},
                'power_range': (50, 70),  # dBm
                'typical_height_range': (10, 50),
                'height_distribution': 'ground',
                'path_loss_factor': 1.8,
                'coverage_radius': 1000,
                'description': '卫星通信地面接收站',
                'color': '#FFA07A',  # 橙色
                'power_semantics': 'eirp_equivalent_dbm',
                'calibration': {
                    'eirp_range_dbm': (50.0, 70.0),
                    'eirp_range_dbw': (20.0, 40.0),  # dBW = dBm - 30
                    'eirp_mapping_dbw': 'P_EIRP[dBW] = 10log10(P_BUC[W]) + G_t[dBi] - L_t[dB]',
                    'unit_note': '本工程内部使用 dBm；若参考资料以 dBW 给出 EIRP，请用 dBm = dBW + 30 换算',
                    'notes': [
                        '该类型在仿真中作为“等效辐射源/干扰源”使用；不区分 conducted power 与天线增益，统一折算进 EIRP-equivalent',
                        '50–70 dBm (=20–40 dBW) 是“Ku 地面端等效EIRP”保守区间，用于覆盖不同终端/BUC 等级与系统余量'
                    ],
                    'sources': {
                        'eirp_range_dbw': '待补齐：便携 Ku 终端规格（EIRP, dBW）+ BUC 规格（W）双锚点；将终端 EIRP(dBW) 与 BUC(W) 通过上式统一到 EIRP-equivalent 口径'
                    }
                }
            }
        }

        # 干扰源类型映射
        self.type_mapping = {
            0: 'wifi_2_4g',
            1: 'wifi_5_8g',
            2: 'cellular_4g',
            3: 'cellular_5g',
            4: 'gnss_jammer',
            5: 'industrial_device',
            6: 'satellite_ground',
            7: 'cellular_ue_ul',
        }

        # 高度分布类型说明
        self.height_distribution_types = {
            'ground': '地面安装',
            'building_low': '建筑物低层(1-5层)',
            'building_mid': '建筑物中层(6-15层)',
            'building_top': '建筑物顶层'
        }

    def get_config(self, type_id: int) -> Dict:
        """获取指定类型的配置"""
        type_name = self.type_mapping.get(type_id, 'wifi_2_4g')
        return self.interference_configs[type_name]

    def get_all_types(self) -> List[str]:
        """获取所有类型名称"""
        return [config['name'] for config in self.interference_configs.values()]

    def get_type_id_by_name(self, name: str) -> int:
        """根据名称获取类型ID"""
        for type_id, type_name in self.type_mapping.items():
            if self.interference_configs[type_name]['name'] == name:
                return type_id
        return 0

    def get_allowed_semantics(self, type_id: int) -> List[str]:
        """
        每类干扰源允许出现的“场景语义集合” Γ(type)。
        这是场景一致性约束层，不改变点过程/功率生成逻辑，只做条件重采样/投影。
        """
        type_name = self.type_mapping.get(int(type_id), 'wifi_2_4g')

        # 约定语义标签：street/sidewalk/ground/roof/indoor/air/pole/tower
        if type_name in ('wifi_2_4g', 'wifi_5_8g'):
            return ['indoor']
        if type_name == 'cellular_4g':
            return ['roof', 'tower']
        if type_name == 'cellular_5g':
            return ['roof', 'pole', 'tower']
        if type_name == 'cellular_ue_ul':
            return ['street', 'sidewalk', 'indoor', 'air']
        if type_name == 'gnss_jammer':
            return ['street', 'sidewalk']
        if type_name == 'industrial_device':
            return ['indoor', 'ground', 'sidewalk']
        if type_name == 'satellite_ground':
            return ['ground', 'roof']
        return ['ground']

    def get_duty_cycle(self, type_id: int) -> float:
        """burstiness：仅用于 WiFi 等间歇式业务，返回 [0,1] 占空比。"""
        type_name = self.type_mapping.get(int(type_id), 'wifi_2_4g')
        if type_name in ('wifi_2_4g', 'wifi_5_8g'):
            return 0.25
        return 1.0


# 复用配置实例，避免在热点路径中重复构造大字典
INTERFERENCE_SOURCE_CONFIG = InterferenceSourceConfig()


class Building:
    """建筑物类"""

    def __init__(self, x: float, y: float, height: float, width: float, length: float):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.length = length
        self.color = '#45B7D1'

    def footprint_area_m2(self) -> float:
        return float(self.width) * float(self.length)

    def volume_m3(self) -> float:
        return self.footprint_area_m2() * float(self.height)

    def floors(self, floor_height_m: float) -> int:
        floor_h = max(float(floor_height_m), 0.5)
        return max(1, int(math.floor(float(self.height) / floor_h)))

    def floor_area_m2(self, floor_height_m: float) -> float:
        return self.footprint_area_m2() * float(self.floors(floor_height_m))

    def contains_point(self, point_x: float, point_y: float) -> bool:
        """判断点是否在建筑物平面范围内"""
        return (self.x - self.width / 2 <= point_x <= self.x + self.width / 2 and
                self.y - self.length / 2 <= point_y <= self.y + self.length / 2)

    def intersects_segment_3d(self, p0: np.ndarray, p1: np.ndarray, eps: float = 1e-6) -> bool:
        """
        判断 3D 线段 p0->p1 是否与建筑物轴对齐长方体相交（建筑物底面在 z=0，高度为 self.height）。

        说明：
        - 建筑物在平面上是轴对齐矩形，空间体为 AABB。
        - 为避免“端点恰好落在边界上”造成误判，使用 eps 排除 t=0/1 的纯端点触碰。
        """
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        d = p1 - p0

        xmin = float(self.x - self.width / 2)
        xmax = float(self.x + self.width / 2)
        ymin = float(self.y - self.length / 2)
        ymax = float(self.y + self.length / 2)
        zmin = 0.0
        zmax = float(self.height)

        tmin = 0.0
        tmax = 1.0

        for axis, (bmin, bmax) in enumerate(((xmin, xmax), (ymin, ymax), (zmin, zmax))):
            if abs(float(d[axis])) < 1e-12:
                # 与该轴平行，若起点不在 slab 内则无相交
                if float(p0[axis]) < bmin or float(p0[axis]) > bmax:
                    return False
                continue

            inv = 1.0 / float(d[axis])
            t1 = (bmin - float(p0[axis])) * inv
            t2 = (bmax - float(p0[axis])) * inv
            t_enter = min(t1, t2)
            t_exit = max(t1, t2)
            tmin = max(tmin, t_enter)
            tmax = min(tmax, t_exit)
            if tmax < tmin:
                return False

        # 排除纯端点触碰：要求相交区间落在 (eps, 1-eps) 内
        return (tmax > eps) and (tmin < 1.0 - eps)

    def get_suitable_interference_heights(self, interference_type: str) -> List[float]:
        """根据干扰源类型获取合适的安装高度"""
        height_distribution = {
            'ground': [2],  # 地面
            'building_low': [self.height * 0.2, self.height * 0.4],  # 低层
            'building_mid': [self.height * 0.5, self.height * 0.7],  # 中层
            'building_top': [self.height * 0.9, self.height + 5]  # 顶层+天线
        }

        type_configs = INTERFERENCE_SOURCE_CONFIG.interference_configs
        for _, config_info in type_configs.items():
            if config_info['name'] == interference_type:
                dist_type = config_info['height_distribution']
                return height_distribution.get(dist_type, [self.height * 0.5])

        return [self.height * 0.5]  # 默认中层

    def get_random_position_on_building(self, height_offset: float = 0) -> Tuple[float, float, float]:
        """获取建筑物上的随机位置"""
        # 在建筑物范围内随机位置
        pos_x = np.random.uniform(self.x - self.width / 2, self.x + self.width / 2)
        pos_y = np.random.uniform(self.y - self.length / 2, self.y + self.length / 2)

        # 加上高度偏移（天线高度等）
        total_height = self.height + height_offset

        return pos_x, pos_y, total_height


class InterferenceSource:
    """干扰源类"""

    def __init__(self, x: float, y: float, height: float,
                 type_id: int, power: float, building=None, is_ground: bool = False,
                 building_idx: Optional[int] = None,
                 is_indoor: bool = False,
                 indoor_floor: Optional[int] = None,
                 indoor_floors_total: Optional[int] = None,
                 indoor_distance_m: Optional[float] = None,
                 skip_height_clip: bool = False):
        self.x = x
        self.y = y
        self.height = height
        self.type_id = type_id
        self.power = power
        self.building = building  # 关联的建筑物
        self.building_idx = building_idx  # 关联建筑物在列表中的索引（用于加速遮挡判断）
        self.is_ground = is_ground  # 是否为地面干扰源
        self.is_indoor = bool(is_indoor)
        self.indoor_floor = indoor_floor
        self.indoor_floors_total = indoor_floors_total
        self.indoor_distance_m = indoor_distance_m

        # 获取配置信息
        self.config = INTERFERENCE_SOURCE_CONFIG.get_config(type_id)
        self.name = self.config['name']
        self.frequency = self.config['frequency']
        self.color = self.config['color']
        # 保持原始含义：配置中的“路径损耗系数”（线性域因子）
        self.path_loss_factor = float(self.config.get('path_loss_factor', 1.0))
        self.coverage_radius = self.config['coverage_radius']

        # 调整功率在合理范围内
        min_power, max_power = self.config['power_range']
        self.power = np.clip(power, min_power * 0.8, max_power * 1.2)

        # 调整高度在典型范围内
        min_height, max_height = self.config['typical_height_range']

        # 如果是地面干扰源，高度固定为0
        if is_ground:
            self.height = 0.0
        elif skip_height_clip:
            self.height = float(max(0.0, height))
        else:
            self.height = float(np.clip(height, min_height, max_height))

    def calculate_path_loss(self, distance: float, drone_height: float, is_blocked: bool = False) -> float:
        """计算路径损耗（dB）"""
        if distance < 1:
            distance = 1.0

        # 自由空间路径损耗（保持原公式）
        fspl_db = float(20.0 * np.log10(distance) + 20.0 * np.log10(self.frequency) - 147.55)

        # 考虑高度的额外损耗
        height_factor = self._calculate_height_factor(drone_height)

        # 将原先的“乘法因子”改为 dB 域加法：
        # total_loss_db = FSPL(dB) + 10*log10(path_loss_factor * height_factor) + L_obstruction(dB)
        factor = float(self.path_loss_factor) * float(height_factor)
        extra_loss_db = float(10.0 * np.log10(max(factor, 1e-12)))

        # 遮挡衰减（保持原来的 0.7 常数，但作为“功率衰减因子”使用）：
        # P_rx *= 0.7  <=>  PL += -10*log10(0.7)
        obstruction_attenuation = 0.7 if is_blocked else 1.0
        obstruction_loss_db = float(-10.0 * np.log10(max(obstruction_attenuation, 1e-12)))

        total_loss_db = float(fspl_db + extra_loss_db + obstruction_loss_db)

        return total_loss_db

    def _calculate_height_factor(self, drone_height: float) -> float:
        """计算高度因子"""
        # 如果是地面干扰源（高度为0），使用不同的高度因子
        if self.is_ground:
            # 地面干扰源对低空无人机影响更大
            if drone_height < 50:
                return 0.9  # 地面干扰对低空影响较大
            elif drone_height < 100:
                return 1.1
            else:
                return 1.3  # 地面干扰对高空影响较小

        # 非地面干扰源的原有逻辑
        min_h, max_h = self.config['typical_height_range']

        if drone_height < min_h:
            # 无人机低于干扰源典型高度范围
            return 1.2  # 增加损耗
        elif drone_height > max_h:
            # 无人机高于干扰源典型高度范围
            return 0.8  # 减少损耗（高空视距更好）
        else:
            # 在典型范围内
            normalized = (drone_height - min_h) / (max_h - min_h)
            return 1.0 - normalized * 0.4

    def get_effective_power(self, drone_pos: np.ndarray, buildings: Optional[List[Building]] = None) -> float:
        """计算对无人机的有效干扰功率"""
        # 计算距离
        distance = np.sqrt((self.x - drone_pos[0]) ** 2 +
                           (self.y - drone_pos[1]) ** 2 +
                           (self.height - drone_pos[2]) ** 2)

        # 计算遮挡：线段是否穿过任意建筑体（排除干扰源自身所在建筑）
        is_blocked = False
        if buildings:
            p0 = np.array([self.x, self.y, self.height], dtype=float)
            p1 = np.asarray(drone_pos, dtype=float)
            for b in buildings:
                if (self.building is not None) and (b is self.building):
                    continue
                if b.intersects_segment_3d(p0, p1):
                    is_blocked = True
                    break

        # 计算路径损耗（dB）
        path_loss_db = self.calculate_path_loss(float(distance), float(drone_pos[2]), is_blocked=is_blocked)

        # 有效功率 = 发射功率 - 路径损耗
        effective_power_dbm = float(self.power - path_loss_db)

        # 考虑距离衰减（超出覆盖范围衰减）
        if distance > self.coverage_radius and self.coverage_radius > 0:
            excess_ratio = float(distance / self.coverage_radius)
            # 以 1/d^2 形式额外衰减：P(dBm) -= 20log10(excess_ratio)
            effective_power_dbm -= float(20.0 * np.log10(max(excess_ratio, 1e-6)))

        return float(effective_power_dbm)

    def get_info(self) -> Dict:
        """获取干扰源信息"""
        pm = getattr(self, 'profile_meta', None)
        power_semantics = pm.get('power_semantics', 'eirp_equivalent_dbm') if isinstance(pm, dict) else 'eirp_equivalent_dbm'
        return {
            'name': self.name,
            'type_id': self.type_id,
            'position': [float(self.x), float(self.y), float(self.height)],
            'power_dbm': float(self.power),
            'raw_power_dbm': float(getattr(self, 'raw_power_dbm', self.power)),
            'profile_key': getattr(self, 'profile_key', None),
            'profile_meta': getattr(self, 'profile_meta', None),
            'power_semantics': str(power_semantics),
            'frequency_ghz': float(self.frequency / 1e9),
            'building_associated': self.building is not None,
            'is_ground': self.is_ground,
            'is_indoor': self.is_indoor,
            'indoor_floor': self.indoor_floor,
            'indoor_floors_total': self.indoor_floors_total,
            'semantic': getattr(self, 'semantic', None),
            'activity': float(getattr(self, 'activity', 1.0)),
            'color': self.color
        }


class DroneCommProblem(ea.Problem):
    """无人机通信性能劣化问题定义"""

    ATG_ENV_PRESETS = {
        # 常用环境参数（Optimal LAP / Al-Hourani 模型常见取值）
        # C,B 对应 P(LoS)=1/(1+C·exp(-B(θ-C)))（θ:度）
        'suburban': {'eta_los_db': 0.1, 'eta_nlos_db': 21.0, 'los_C': 4.88, 'los_B': 0.43},
        'urban': {'eta_los_db': 1.0, 'eta_nlos_db': 20.0, 'los_C': 9.61, 'los_B': 0.16},
        'dense_urban': {'eta_los_db': 1.6, 'eta_nlos_db': 23.0, 'los_C': 12.08, 'los_B': 0.11},
        'highrise_urban': {'eta_los_db': 2.3, 'eta_nlos_db': 34.0, 'los_C': 27.23, 'los_B': 0.08},
    }


    # EARTH reference (Section 4.3 / Table 4-3) for BS power model
    EARTH_BS_PARAMS = {
        'macro': {'p_max_w': 20.0, 'n_trx': 6, 'p0_w': 130.0, 'delta_p': 4.7, 'p_sleep_w': 75.0},
        'micro': {'p_max_w': 6.3, 'n_trx': 2, 'p0_w': 56.0, 'delta_p': 2.6, 'p_sleep_w': 39.0},
        'pico': {'p_max_w': 0.13, 'n_trx': 2, 'p0_w': 6.8, 'delta_p': 4.0, 'p_sleep_w': 4.3},
        'femto': {'p_max_w': 0.05, 'n_trx': 2, 'p0_w': 4.8, 'delta_p': 8.0, 'p_sleep_w': 2.9},
    }

    # Al-Hourani 2014: Surface polynomial coefficients mapping (αβ, γ) -> a,b in Eq.(5)/(6)
    # z = Σ_{j=0..3} Σ_{i=0..3-j} C_ij * (αβ)^i * γ^j
    # Table I: coefficients for a
    LOS_A_COEFFS: Dict[int, Dict[int, float]] = {
        0: {0: 9.34e-01, 1: 2.30e-01, 2: -2.25e-02, 3: 1.86e-05},
        1: {0: 1.97e-02, 1: 2.44e-03, 2: 6.58e-06},
        2: {0: -1.24e-04, 1: -3.34e-06},
        3: {0: 2.73e-07},
    }

    # Table II: coefficients for b
    LOS_B_COEFFS: Dict[int, Dict[int, float]] = {
        0: {0: 1.17e+00, 1: -7.56e-02, 2: 1.98e-03, 3: -1.78e-05},
        1: {0: -5.79e-03, 1: 1.81e-04, 2: -1.65e-06},
        2: {0: 1.73e-05, 1: -2.02e-07},
        3: {0: -2.00e-08},
    }

    def __init__(self, user_params: Dict):
        name = '无人机通信性能劣化优化'
        m = 1  # 目标维度（单目标优化）
        maxormins = [-1]  # -1表示最大化适应度（最大化劣化程度）
        self.user_params = dict(user_params)

        # 用户参数
        self.num_drones = int(user_params['num_drones'])  # 无人机数量
        self.num_stations = int(user_params['num_stations'])  # 地面站数量
        self.area_size = float(user_params['area_size'])  # 区域大小
        self.drone_height_max = float(user_params['drone_height_max'])  # 无人机最大高度
        self.building_height_max = float(user_params['building_height_max'])  # 建筑物最大高度
        self.building_density = float(user_params['building_density'])  # 建筑物密度
        self.drone_speed_max = float(user_params['drone_speed_max'])  # 无人机最大速度

        # EARTH time-domain BS power model (dynamic energy)
        self.energy_model = str(user_params.get('energy_model', 'earth')).strip().lower()
        self.bs_energy_enabled = bool(user_params.get('bs_energy_enabled', self.energy_model == 'earth'))
        self.bs_type = str(user_params.get('bs_type', 'micro')).strip().lower()
        self.bs_params_override = dict(user_params.get('bs_params_override', {}))
        self.bs_capacity_max = float(user_params.get('bs_capacity_max', user_params.get('capacity_max', 100.0)))
        self.bs_users_total = float(user_params.get('bs_users_total', user_params.get('users_total', float(self.num_stations) * 20.0)))
        self.bs_users_per_station = float(user_params.get('bs_users_per_station', self.bs_users_total / max(float(self.num_stations), 1.0)))
        self.bs_time_origin_s = float(user_params.get('bs_time_origin_s', 0.0))
        self.bs_sleep_enabled = bool(user_params.get('bs_sleep_enabled', True))
        self.bs_load_floor = float(user_params.get('bs_load_floor', 0.0))
        self.include_bs_energy_in_ee = bool(user_params.get('include_bs_energy_in_ee', True))
        self.bs_traffic_profile_hourly = user_params.get('bs_traffic_profile_hourly', None)
        self.bs_traffic_profile_points = user_params.get('bs_traffic_profile_points', None)
        # 无人机水平位置约束（避免落入建筑内部）
        self.drone_xy_avoid_buildings = bool(user_params.get('drone_xy_avoid_buildings', True))
        self.drone_xy_snap_eps_m = float(user_params.get('drone_xy_snap_eps_m', 0.5))
        self.drone_xy_project_max_iter = int(user_params.get('drone_xy_project_max_iter', 8))

        # ITU 城市统计参数（用于建筑分布与 LoS 概率）
        # α：建筑覆盖率（0~1），β：建筑密度（buildings/km^2），γ：建筑高度 Rayleigh 分布尺度参数（m）
        # 必须在生成建筑物之前初始化
        self.itu_city = {
            'alpha': float(user_params.get('itu_alpha', self.building_density)),
            'beta_km2': float(user_params.get('itu_beta', 300.0)),
            'gamma_m': float(user_params.get('itu_gamma', max(5.0, self.building_height_max / 2.0)))
        }

        # 固定参数
        self.num_interference = int(user_params.get('num_controlled_interference', 8))  # 受控（染色体）干扰源数量
        self.communication_freq = 2.4e9  # 无人机通信频率
        self.light_speed = 3e8  # 光速

        # 背景干扰源：用 3D PPP 生成大量“城市干扰场”（不进入染色体维度）
        self.background_interference_enabled = bool(user_params.get('background_interference_enabled', True))
        self.background_interference_density_scale = float(user_params.get('background_interference_density_scale', 1.0))
        # 2D/3D PPP 强度口径：
        # - '2d'：base_lambda_per_km2 直接表示 λ（sources/km^2）
        # - '3d'：可额外提供 bg_rho_m3_{type}（sources/m^3），并用 ρ=λ/Z 换算，Z=background_interference_box_height_m
        self.background_interference_density_mode = str(user_params.get('background_interference_density_mode', '2d')).strip().lower()
        self.background_interference_box_height_m = float(user_params.get('background_interference_box_height_m', self.building_height_max))
        self.background_interference_max_sources = int(user_params.get('background_interference_max_sources', 500))
        self.max_sources_to_plot = int(user_params.get('max_sources_to_plot', 60))

        # 城市场景“标签层”（street/sidewalk/roof/indoor/air/facade/pole...）
        # 只做约束/投影，不推翻原始随机生成公式
        self.scene_consistency_mode = str(user_params.get('scene_consistency_mode', 'resample')).strip().lower()
        self.scene_consistency_max_attempts = int(user_params.get('scene_consistency_max_attempts', 12))
        self.semantic_road_spacing_m = float(user_params.get('semantic_road_spacing_m', 0.0))  # 0 表示自动从 ITU 统计量推断
        self.semantic_road_width_m = float(user_params.get('semantic_road_width_m', 20.0))
        self.semantic_sidewalk_width_m = float(user_params.get('semantic_sidewalk_width_m', 6.0))
        self.semantic_facade_band_m = float(user_params.get('semantic_facade_band_m', 2.0))
        self.semantic_pole_snap_m = float(user_params.get('semantic_pole_snap_m', 3.0))

        # 室内/楼层参数（用于“地面+楼层+屋顶”的 3D 分布）
        self.floor_height_m = float(user_params.get('floor_height_m', 3.0))
        self.indoor_device_height_m = float(user_params.get('indoor_device_height_m', 1.5))

        # ITU P.1238（室内）简化项：L = 20log(f_MHz) + Nlog(d_m) + Lf(n) - 28
        self.p1238_enabled = bool(user_params.get('p1238_enabled', True))
        self.p1238_N = float(user_params.get('p1238_N', 30.0))
        self.p1238_d_in_m = float(user_params.get('p1238_d_in_m', 10.0))

        # ITU P.2109（BEL）参数化项：仅启用时生效（默认不启用，避免硬编码系数）
        self.p2109_bel_enabled = bool(user_params.get('p2109_bel_enabled', False))
        self.p2109_bel_r = user_params.get('p2109_bel_r', None)
        self.p2109_bel_s = user_params.get('p2109_bel_s', None)
        self.p2109_bel_t = user_params.get('p2109_bel_t', None)
        self.p2109_bel_u = user_params.get('p2109_bel_u', None)
        self.p2109_bel_li_db = user_params.get('p2109_bel_li_db', None)
        self.p2109_bel_lg_clip_db = float(user_params.get('p2109_bel_lg_clip_db', 0.0))

        # 统一“强度/损耗”劣化指标选择：'power_ratio'(默认，原逻辑) 或 'cdi'（强度/损耗合成）
        # 干扰源配置（必须在生成固定场景元素/背景PPP前就绪）
        self.interference_config = INTERFERENCE_SOURCE_CONFIG

        # 生成固定场景元素
        np.random.seed(42)  # 固定随机种子确保可重复性
        self._generate_fixed_scene_elements()

        # 决策变量定义
        # 染色体包括：
        # 1. 无人机高度 (num_drones 个)
        # 2. 无人机速度 (num_drones 个，0到最大速度)
        # 3. 无人机水平位置 (num_drones * 2 个)
        # 4. 干扰源类型 (num_interference 个)
        # 5. 干扰源功率 (num_interference 个)
        # 6. 干扰源位置类型 (num_interference 个，0=地面，1=建筑物)
        # 7. 干扰源在建筑物上的位置偏移或地面位置 (num_interference * 2 个)

        # 计算维度
        dim = (self.num_drones * 4 +  # 无人机高度、速度、水平位置(x,y)
               self.num_interference * 5)  # 干扰源类型、功率、位置类型、位置(x,y)

        # 变量类型
        var_types = ([0] * self.num_drones +  # 无人机高度: 连续
                     [0] * self.num_drones +  # 无人机速度: 连续
                     [0] * self.num_drones * 2 +  # 无人机水平位置: 连续
                     [1] * self.num_interference +  # 干扰源类型: 离散
                     [0] * self.num_interference +  # 干扰源功率: 连续
                     [1] * self.num_interference +  # 干扰源位置类型: 离散 (0=地面, 1=建筑物)
                     [0] * self.num_interference * 2)  # 位置: 连续

        # 变量下界
        lb = ([10] * self.num_drones +  # 无人机高度下限
              [0.0] * self.num_drones +  # 无人机速度下限（允许悬停 V=0）
              [0.0] * self.num_drones * 2 +  # 无人机水平位置下限
              [0] * self.num_interference +  # 干扰源类型下限
              [0] * self.num_interference +  # 干扰源功率下限
              [0] * self.num_interference +  # 干扰源位置类型下限
              [0] * self.num_interference * 2)  # 位置范围 [0, area_size]

        # 变量上界
        ub = ([self.drone_height_max] * self.num_drones +  # 无人机高度上限
              [self.drone_speed_max] * self.num_drones +  # 无人机速度上限
              [self.area_size] * self.num_drones * 2 +  # 无人机水平位置上限
              [7] * self.num_interference +  # 干扰源类型上限（0-7）
              [70] * self.num_interference +  # 干扰源功率上限
              [1] * self.num_interference +  # 干扰源位置类型上限 (0-1)
              [self.area_size] * self.num_interference * 2)  # 位置范围

        # 验证维度一致性
        assert len(var_types) == dim, f"var_types长度{len(var_types)}与dim{dim}不匹配"
        assert len(lb) == dim, f"lb长度{len(lb)}与dim{dim}不匹配"
        assert len(ub) == dim, f"ub长度{len(ub)}与dim{dim}不匹配"

        # 调用父类构造方法
        super().__init__(name, m, maxormins, dim, var_types, lb, ub)

        # 权重参数
        self.metric_weights = {
            'power_intensity_change': 0.7,
            'speed_energy_efficiency': 0.3
        }

        # 通信劣化评估模型：统一使用 margin（功率裕量）模型
        # 注：legacy/CDI 分支已停用（保留代码以便回溯/对照）
        # NOTE: 统一使用 margin（功率裕量）模型；CDI/legacy 分支停用。
        self.degradation_model = 'margin'

        # 功率域参数：噪声功率 Ny = N0·B·F（线性域 mW 累加）
        self.bandwidth_hz = float(user_params.get('bandwidth_hz', 20e6))
        self.noise_figure_db = float(user_params.get('noise_figure_db', 7.0))
        self.thermal_noise_dbm_hz = float(user_params.get('thermal_noise_dbm_hz', -174.0))
        # 功率裕量门限/斜率：M[dB]=P_rx[dBm]-P_IN[dBm]，Outage=1{M<θ_M}
        self.margin_threshold_db = float(user_params.get('margin_threshold_db', 0.0))
        self.margin_slope_db = float(user_params.get('margin_slope_db', 3.0))

        # CDI 参数（已停用，仅为兼容旧配置文件）
        # CDI 参数已停用（保留字段以兼容旧的 params JSON）。
        self.cdi_pl_max_db = float(user_params.get('cdi_pl_max_db', 125.0))
        self.cdi_w_link = float(user_params.get('cdi_w_link', 1.0))
        self.cdi_w_int = float(user_params.get('cdi_w_int', 1.0))
        # g(I) = 10log10(1 + I/I_ref)，I_ref 用 mW（默认 -30 dBm = 1e-3 mW）
        self.cdi_int_ref_mw = float(user_params.get('cdi_int_ref_mw', 1e-3))
        # 将 CDI 映射到 [0,1] 的尺度（已停用）
        self.cdi_scale_db = float(user_params.get('cdi_scale_db', 20.0))

        # 可选：在 shot-noise 中加入小尺度衰落 h_i（Rayleigh 功率增益 ~ Exp(1)），并用稳定伪随机保证可复现
        self.interference_fading_enabled = bool(user_params.get('interference_fading_enabled', False))
        self.signal_fading_enabled = bool(user_params.get('signal_fading_enabled', False))

        # Interference aggregation: Nakagami-m moment matching vs linear sum
        self.interference_aggregation_model = str(user_params.get('interference_aggregation_model', 'nakagami')).strip().lower()
        self.nakagami_m_default = float(user_params.get('nakagami_m_default', 1.0))
        self.nakagami_m_by_type = dict(user_params.get('nakagami_m_by_type', {}))
        self.nakagami_m_los = user_params.get('nakagami_m_los', None)
        self.nakagami_m_nlos = user_params.get('nakagami_m_nlos', None)
        self.nakagami_sample_mode = str(user_params.get('nakagami_sample_mode', 'sample')).strip().lower()
        self.nakagami_mu4_mode = str(user_params.get('nakagami_mu4_mode', 'fast')).strip().lower()
        self.nakagami_m_min = float(user_params.get('nakagami_m_min', 0.5))
        self.nakagami_m_max = float(user_params.get('nakagami_m_max', 20.0))

        # 大纲式“功率主导”最终评估参数（概率型输出）
        self.objective_model = str(user_params.get('objective_model', 'power_score')).strip().lower()
        self.succ_threshold_dbm = float(user_params.get('succ_threshold_dbm', -90.0))  # θ_p：有用接收功率阈值
        self.int_threshold_dbm = float(user_params.get('int_threshold_dbm', -80.0))    # 干扰功率阈值
        self.pl_max_db = float(user_params.get('pl_max_db', 125.0))                    # PL_max：路径损耗阈值
        self.power_score_weights = {
            'w1_succ': float(user_params.get('w1_succ', 1.0)),
            'w2_int': float(user_params.get('w2_int', 1.0)),
            'w3_pl': float(user_params.get('w3_pl', 1.0)),
        }

        # 可选：U2U 链路纳入同构评估（A2G/U2U 同一套几何+LoS+功率预算，只差高度项）
        self.u2u_enabled = bool(user_params.get('u2u_enabled', False))
        self.u2u_tx_power_dbm = float(user_params.get('u2u_tx_power_dbm', 23.0))
        self.u2u_pairing = str(user_params.get('u2u_pairing', 'all')).strip().lower()  # all | nearest

        # 可选：离散时间 TDMA + EE 评估（不改变默认静态场景；仅在 tdma_enabled=True 时启用）
        self.tdma_enabled = bool(user_params.get('tdma_enabled', False))
        # 总评分（大纲 E3）：D_total = w1 D_comm + w2 D_thr + w3 D_EE + w4 D_penalty
        self.outline_weights = {
            'w_comm': float(user_params.get('outline_w_comm', 1.0)),
            'w_thr': float(user_params.get('outline_w_thr', 0.0)),
            'w_ee': float(user_params.get('outline_w_ee', 0.0)),
            'w_penalty': float(user_params.get('outline_w_penalty', 0.0)),
        }

        self.power_model = {
            'comm_tx_power_dbm': 20.0,
            'baseline_distance_m': 100.0,
            'nlos_excess_loss_db': 20.0,
            'power_change_scale_db': 15.0,
            'speed_power_beta': 1.0,

            # 旋翼无人机推进功率模型（直线水平飞行）
            # 参考：Y. Zeng et al., IEEE TWC 2019（rotary-wing UAV power model）
            'rotor_P0_w': 79.86,      # blade profile power at hover (W)
            'rotor_Pi_w': 88.63,      # induced power at hover (W)
            'rotor_U_tip_mps': 120.0, # rotor blade tip speed (m/s)
            'rotor_v0_mps': 4.03,     # mean rotor induced velocity in hover (m/s)
            'rotor_d0': 0.6,          # fuselage drag ratio
            'air_rho': 1.225,         # air density (kg/m^3)
            'rotor_s': 0.05,          # rotor solidity
            'rotor_A_m2': 0.503       # rotor disc area (m^2)
        }

        # 空对地（ATG）LoS/NLoS 路径损耗模型（Optimal LAP Altitude for Maximum Coverage）
        # PL_ξ = FSPL + η_ξ
        # Λ = P(LoS)·PL_LoS + (1-P(LoS))·PL_NLoS
        # P(LoS,θ) = 1 / (1 + C·exp(-B(θ-C)))，θ为仰角（度）
        self.atg_model = {
            'eta_los_db': 1.0,
            'eta_nlos_db': 20.0,
            # 论文/图片中常用记号为 C、B；此处保留历史字段名 los_a/los_b
            'los_a': float(user_params.get('los_C', user_params.get('los_a', 9.61))),
            'los_b': float(user_params.get('los_B', user_params.get('los_b', 0.16))),
            # LoS 概率模型选择：
            # - 'sigmoid'：Eq.(5)/(2)，使用 los_a/los_b（默认）
            # - 'sigmoid_fit'：Eq.(6) + Table I/II，由 (αβ,γ) 自动拟合计算 a,b
            # - 'itu_product'：Eq.(4) 原始乘积形式
            'los_model': str(user_params.get('los_model', 'sigmoid_fit')).strip().lower()
        }

        # LoS 指示量模式（“有地图→几何/射线追踪；无地图→概率模型”）
        # - deterministic: 只要线段不穿建筑 => LoS=1（否则0）
        # - probabilistic: 始终用概率模型 P_LoS(θ)
        # - deterministic_if_map: buildings 不为空时用 deterministic；否则用 probabilistic（默认）
        self.los_indicator_mode = str(user_params.get('los_indicator_mode', 'deterministic_if_map')).strip().lower()

        # 频谱耦合/邻频泄漏（用于把不同频段干扰源纳入同一 SINR/功率叠加）
        # - acir: 近似的 ACIR(Δf) -> 耦合系数，满足“同频=1，邻频<1”
        # - lorentzian: 兼容旧实现 1/(1+(Δf/f0)^2)
        self.spectral_mode = str(user_params.get('spectral_mode', 'acir')).strip().lower()
        self.spectral_model = {
            'acir_db_at_1bw': float(user_params.get('acir_db_at_1bw', 30.0)),            # Δf=BW 时的抑制(dB)
            'acir_slope_db_per_decade': float(user_params.get('acir_slope_db_per_decade', 20.0)),  # 频偏每十倍增加的抑制(dB)
            'acir_db_max': float(user_params.get('acir_db_max', 80.0)),
            'lorentz_f0_hz': float(user_params.get('lorentz_f0_hz', 1e8)),
            'acir_db_at_1bw_by_type': dict(user_params.get('acir_db_at_1bw_by_type', {})),
            'acir_slope_by_type': dict(user_params.get('acir_slope_by_type', {})),
        }

        # 可选：按环境类型一键覆盖 ATG 参数（不改变默认行为；仅当传入 env_type 时生效）
        env_type = str(user_params.get('env_type', '')).strip().lower()
        if env_type in self.ATG_ENV_PRESETS:
            preset = self.ATG_ENV_PRESETS[env_type]
            self.atg_model['eta_los_db'] = float(preset['eta_los_db'])
            self.atg_model['eta_nlos_db'] = float(preset['eta_nlos_db'])
            self.atg_model['los_a'] = float(preset['los_C'])
            self.atg_model['los_b'] = float(preset['los_B'])

        # 可选：按 Table I/II 由 ITU 参数拟合得到 a,b（与论文 Eq.(6) 一致）
        has_manual_sigmoid_params = ('los_C' in user_params) or ('los_B' in user_params) or ('los_a' in user_params) or ('los_b' in user_params)
        if self.atg_model['los_model'] == 'sigmoid_fit' and env_type not in self.ATG_ENV_PRESETS and not has_manual_sigmoid_params:
            fitted_a, fitted_b = self._fit_sigmoid_params_from_itu()
            self.atg_model['los_a'] = fitted_a
            self.atg_model['los_b'] = fitted_b

        # 预计算常量：避免在每条链路上重复计算
        self._comm_tx_power_dbm = float(self.power_model['comm_tx_power_dbm'])
        self._baseline_distance_m = float(self.power_model['baseline_distance_m'])
        self._power_change_scale_db = float(self.power_model['power_change_scale_db'])
        baseline_pl_db = self.calculate_fspl_db(self._baseline_distance_m) + float(self.atg_model['eta_los_db'])
        baseline_rx_dbm = self._comm_tx_power_dbm - baseline_pl_db
        self._baseline_rx_mw = self._dbm_to_mw(baseline_rx_dbm)

    def is_los(self,
               tx_pos: np.ndarray,
               rx_pos: np.ndarray,
               buildings: Optional[List['Building']] = None,
               ignore_building_idx: Optional[int] = None) -> bool:
        """
        统一 LoS 判定接口（确定性）：线段 tx->rx 是否与任意建筑棱柱体相交。
        - 有 buildings：用几何遮挡作为 LoS 指示量
        - 无 buildings：视为 LoS（交给概率模型的 fallback 做统计口径）
        """
        buildings_list = buildings if buildings is not None else []
        if not buildings_list:
            return True
        tx_pos = np.asarray(tx_pos, dtype=float)
        rx_pos = np.asarray(rx_pos, dtype=float)
        if buildings_list is self.buildings:
            blocked = self._is_segment_blocked_by_buildings(tx_pos, rx_pos, ignore_building_idx=ignore_building_idx)
        else:
            blocked = self._is_segment_blocked_by_buildings_naive(tx_pos, rx_pos, buildings_list, ignore_building=None)
        return not bool(blocked)

    def _spectral_coupling_factor(self,
                                  tx_freq_hz: float,
                                  rx_freq_hz: float,
                                  rx_bw_hz: float,
                                  interferer_type_id: Optional[int] = None) -> float:
        """
        频谱耦合系数 η(Δf)∈(0,1]：
        - 同频（|Δf|<=BW/2）: η=1
        - 邻频/异频：用简化 ACIR(Δf) -> 10^(-ACIR/10)
        """
        df = abs(float(tx_freq_hz) - float(rx_freq_hz))
        bw = max(float(rx_bw_hz), 1.0)
        if df <= 0.5 * bw:
            return 1.0

        mode = str(getattr(self, 'spectral_mode', 'acir')).strip().lower()
        # 论文同款（Zeng 2019）常用：只建模同频 cochannel 干扰；异频按 0 处理
        if mode in ('cochannel', 'cochannel_only', 'same_channel'):
            return 0.0
        if mode in ('lorentz', 'lorentzian', 'legacy'):
            f0 = max(float(self.spectral_model.get('lorentz_f0_hz', 1e8)), 1.0)
            return float(1.0 / (1.0 + (df / f0) ** 2))

        # acir
        base = float(self.spectral_model.get('acir_db_at_1bw', 30.0))
        slope = float(self.spectral_model.get('acir_slope_db_per_decade', 20.0))
        if interferer_type_id is not None:
            by_type = self.spectral_model.get('acir_db_at_1bw_by_type', {}) or {}
            if str(int(interferer_type_id)) in by_type:
                base = float(by_type[str(int(interferer_type_id))])
            elif int(interferer_type_id) in by_type:
                base = float(by_type[int(interferer_type_id)])
            slope_by = self.spectral_model.get('acir_slope_by_type', {}) or {}
            if str(int(interferer_type_id)) in slope_by:
                slope = float(slope_by[str(int(interferer_type_id))])
            elif int(interferer_type_id) in slope_by:
                slope = float(slope_by[int(interferer_type_id)])

        ratio = max(df / bw, 1.0)
        acir_db = float(base + slope * np.log10(ratio))
        acir_db = float(np.clip(acir_db, 0.0, float(self.spectral_model.get('acir_db_max', 80.0))))
        return float(10.0 ** (-acir_db / 10.0))

    @staticmethod
    def _dbm_to_mw(power_dbm: float) -> float:
        return float(10 ** (power_dbm / 10.0))

    @staticmethod
    def _mw_to_dbm(power_mw: float) -> float:
        power_mw = max(float(power_mw), 1e-15)
        return float(10.0 * np.log10(power_mw))

    def _fit_sigmoid_params_from_itu(self) -> Tuple[float, float]:
        """
        使用 Al-Hourani 2014 的 surface fitting（Eq.(6) + Table I/II）
        由 ITU 城市参数 (α,β,γ) 计算 Sigmoid 的 a,b（θ:度）。

        说明：论文中使用 x=(αβ)（β: buildings/km^2）, y=γ（m）。
        """
        alpha = float(np.clip(self.itu_city['alpha'], 0.01, 0.99))
        beta_km2 = max(float(self.itu_city['beta_km2']), 1e-9)
        gamma = max(float(self.itu_city['gamma_m']), 1e-9)
        x = alpha * beta_km2
        y = gamma

        def eval_poly(coeffs: Dict[int, Dict[int, float]]) -> float:
            z = 0.0
            for j, row in coeffs.items():
                for i, c_ij in row.items():
                    z += float(c_ij) * (x ** int(i)) * (y ** int(j))
            return float(z)

        a = eval_poly(self.LOS_A_COEFFS)
        b = eval_poly(self.LOS_B_COEFFS)

        # 保护性裁剪：避免极端 ITU 参数导致 Sigmoid 失效
        a = float(np.clip(a, 0.1, 50.0))
        b = float(np.clip(b, 1e-4, 5.0))
        return a, b

    def _generate_fixed_scene_elements(self):
        """生成固定的场景元素（建筑物、地面站位置）"""
        # 生成建筑物
        self._generate_buildings()
        self._building_observables_cache = self._compute_building_observables()
        wifi_density = self._estimate_wifi_ap_density(self._building_observables_cache)
        self._wifi_density_cache = wifi_density if wifi_density is not None else {}

        # 生成地面站位置
        self._generate_station_positions()

        # 生成无人机初始水平位置
        self._generate_drone_xy_positions()

        # 构建建筑物空间索引（用于干扰源“落楼”和遮挡加速）
        self._build_building_spatial_index()

        # 生成城市场景“语义标签层”（用于干扰源场景一致性约束/投影）
        self._generate_semantic_layers()

        # 生成背景干扰源（大量、多类型、多高度；不进入染色体）
        self.background_interference_sources = []
        self.background_interference_sources_viz = []
        if self.background_interference_enabled:
            self._generate_background_interference_sources()

    def _generate_buildings(self):
        """生成建筑物"""
        # 按 ITU 城市统计参数生成：
        # - 数量由 β（buildings/km^2）决定
        # - 高度服从 Rayleigh(γ)
        # - 平面占地总面积近似满足覆盖率 α
        max_buildings = 500
        alpha = float(np.clip(self.itu_city['alpha'], 0.01, 0.99))
        beta_km2 = max(float(self.itu_city['beta_km2']), 1e-6)
        beta_m2 = beta_km2 / 1e6

        area_m2 = float(self.area_size ** 2)
        num_buildings = int(beta_m2 * area_m2)
        num_buildings = max(0, min(num_buildings, max_buildings))

        self.buildings = []
        if num_buildings <= 0:
            return

        target_footprint_area = alpha * area_m2
        avg_footprint_area = max(target_footprint_area / num_buildings, 25.0)
        base_side = float(np.sqrt(avg_footprint_area))

        gamma = max(float(self.itu_city['gamma_m']), 1e-6)
        for _ in range(num_buildings):
            # 建筑高度：Rayleigh 分布
            height = float(np.random.rayleigh(scale=gamma))
            height = float(np.clip(height, 5.0, self.building_height_max))

            # 建筑平面尺寸：围绕平均占地面积随机扰动（再裁剪到合理范围）
            width = float(base_side * np.random.uniform(0.6, 1.4))
            length = float((avg_footprint_area / max(width, 1.0)) * np.random.uniform(0.6, 1.4))
            width = float(np.clip(width, 10.0, 80.0))
            length = float(np.clip(length, 10.0, 80.0))

            # 随机位置（确保建筑物完全落在区域内）
            margin = 5.0
            x_min = width / 2 + margin
            x_max = self.area_size - width / 2 - margin
            y_min = length / 2 + margin
            y_max = self.area_size - length / 2 - margin

            x = float(np.random.uniform(x_min, x_max)) if x_min < x_max else float(self.area_size / 2)
            y = float(np.random.uniform(y_min, y_max)) if y_min < y_max else float(self.area_size / 2)

            building = Building(x, y, height, width, length)
            # 便于调试/索引：记录建筑物序号
            building.idx = len(self.buildings)
            self.buildings.append(building)

        # 记录 W,S（便于报告与调参）：W=1000*sqrt(alpha/beta), S=(1000/sqrt(beta))*(1-sqrt(alpha))
        alpha_cov = float(np.clip(self.itu_city['alpha'], 0.01, 0.99))
        beta_km2_val = max(float(self.itu_city['beta_km2']), 1e-6)
        self.building_geom_W_m = float(1000.0 * np.sqrt(alpha_cov / beta_km2_val))
        self.building_geom_S_m = float((1000.0 / np.sqrt(beta_km2_val)) * (1.0 - np.sqrt(alpha_cov)))

    def _compute_building_observables(self) -> Dict:
        buildings = list(getattr(self, 'buildings', []) or [])
        floor_h = max(float(getattr(self, 'floor_height_m', 3.0)), 0.5)
        obs = {
            'floor_height_m': float(floor_h),
            'building_count': int(len(buildings)),
            'total_footprint_m2': 0.0,
            'total_volume_m3': 0.0,
            'total_floor_area_m2': 0.0,
            'avg_footprint_m2': 0.0,
            'avg_floors': 0.0,
            'avg_floor_area_m2': 0.0,
            'per_building': []
        }
        if not buildings:
            return obs

        total_footprint = 0.0
        total_volume = 0.0
        total_floor_area = 0.0
        total_floors = 0.0
        per_building = []
        for idx, b in enumerate(buildings):
            footprint = float(b.footprint_area_m2())
            floors = int(b.floors(floor_h))
            floor_area = float(footprint * float(floors))
            volume = float(footprint * float(b.height))
            per_building.append({
                'id': int(getattr(b, 'idx', idx)),
                'footprint_m2': footprint,
                'volume_m3': volume,
                'floors': int(floors),
                'floor_area_m2': float(floor_area)
            })
            total_footprint += footprint
            total_volume += volume
            total_floor_area += floor_area
            total_floors += float(floors)

        count = float(len(buildings))
        obs.update({
            'total_footprint_m2': float(total_footprint),
            'total_volume_m3': float(total_volume),
            'total_floor_area_m2': float(total_floor_area),
            'avg_footprint_m2': float(total_footprint / count),
            'avg_floors': float(total_floors / count),
            'avg_floor_area_m2': float(total_floor_area / count),
            'per_building': per_building
        })
        return obs

    def _estimate_wifi_ap_density(self, building_obs: Optional[Dict] = None) -> Optional[Dict]:
        model = str(self.user_params.get('wifi_density_model', 'floor_area')).strip().lower()
        if model in ('legacy', 'fixed', 'off', 'none', 'disable'):
            return None

        area_km2 = float((self.area_size / 1000.0) ** 2)
        if area_km2 <= 0.0:
            return None

        if building_obs is None:
            building_obs = self._compute_building_observables()
        total_floor_m2 = float(building_obs.get('total_floor_area_m2', 0.0))
        if total_floor_m2 <= 0.0:
            return None

        def clamp01(value: float, default: float) -> float:
            try:
                v = float(value)
            except Exception:
                v = float(default)
            return float(np.clip(v, 0.0, 1.0))

        def safe_pos(value: float, default: float) -> float:
            try:
                v = float(value)
            except Exception:
                v = float(default)
            if v <= 0.0:
                v = float(default)
            return float(v)

        eta_res = clamp01(self.user_params.get('wifi_eta_res', 0.6), 0.6)
        eta_bus_raw = self.user_params.get('wifi_eta_bus', None)
        if eta_bus_raw is None:
            eta_bus = max(0.0, 1.0 - eta_res)
        else:
            eta_bus = clamp01(eta_bus_raw, max(0.0, 1.0 - eta_res))
            total_share = float(eta_res + eta_bus)
            if total_share > 1.0:
                eta_res = float(eta_res / total_share)
                eta_bus = float(eta_bus / total_share)

        a_dwell = safe_pos(self.user_params.get('wifi_avg_dwell_area_m2', 80.0), 80.0)
        kappa_res = max(0.0, float(self.user_params.get('wifi_res_router_per_dwell', 1.0)))
        p_res = clamp01(self.user_params.get('wifi_res_adoption', 0.8), 0.8)
        n_dwell = float(eta_res * total_floor_m2 / a_dwell)
        n_res = float(kappa_res * p_res * n_dwell)

        business_model = str(self.user_params.get('wifi_business_model', 'coverage_area')).strip().lower()
        bus_detail: Dict[str, float] = {}
        if business_model in ('office_unit', 'office', 'unit', 'b1'):
            a_office = safe_pos(self.user_params.get('wifi_avg_office_area_m2', 120.0), 120.0)
            kappa_bus = max(0.0, float(self.user_params.get('wifi_bus_ap_per_office', 1.0)))
            p_bus = clamp01(self.user_params.get('wifi_bus_adoption', 0.8), 0.8)
            n_office = float(eta_bus * total_floor_m2 / a_office)
            n_bus = float(kappa_bus * p_bus * n_office)
            bus_detail = {
                'business_model': 'office_unit',
                'A_office_m2': float(a_office),
                'kappa_bus': float(kappa_bus),
                'p_bus': float(p_bus),
                'N_office': float(n_office)
            }
        else:
            alpha_bus = clamp01(self.user_params.get('wifi_bus_coverage_ratio', 0.7), 0.7)
            c_ap = safe_pos(self.user_params.get('wifi_ap_coverage_area_m2', 200.0), 200.0)
            a_bus_floor = float(eta_bus * total_floor_m2)
            n_bus = float(alpha_bus * a_bus_floor / c_ap)
            bus_detail = {
                'business_model': 'coverage_area',
                'alpha_bus': float(alpha_bus),
                'C_ap_m2': float(c_ap),
                'A_bus_floor_m2': float(a_bus_floor)
            }

        n_wifi = float(n_res + n_bus)
        lambda_wifi = float(n_wifi / area_km2)
        f_24 = clamp01(self.user_params.get('wifi_band_share_24', 0.6), 0.6)
        lambda_24_base = float(lambda_wifi * f_24)
        lambda_58_base = float(lambda_wifi * (1.0 - f_24))
        p_tx = clamp01(self.user_params.get('wifi_tx_activity_prob', 1.0), 1.0)
        lambda_24 = float(lambda_24_base * p_tx)
        lambda_58 = float(lambda_58_base * p_tx)

        return {
            'model': str(model),
            'area_km2': float(area_km2),
            'eta_res': float(eta_res),
            'eta_bus': float(eta_bus),
            'A_dwell_m2': float(a_dwell),
            'kappa_res': float(kappa_res),
            'p_res': float(p_res),
            'N_dwell': float(n_dwell),
            'N_res': float(n_res),
            'N_bus': float(n_bus),
            'N_wifi': float(n_wifi),
            'lambda_wifi_km2': float(lambda_wifi),
            'lambda_24_base_km2': float(lambda_24_base),
            'lambda_58_base_km2': float(lambda_58_base),
            'lambda_24_km2': float(lambda_24),
            'lambda_58_km2': float(lambda_58),
            'f_24': float(f_24),
            'p_tx': float(p_tx),
            **bus_detail
        }

    def _generate_background_interference_sources(self):
        """生成背景干扰源（2D PPP 投影 + 高度采样，混合地面/屋顶/楼层）。"""
        area_km2 = float((self.area_size / 1000.0) ** 2)

        # 每类干扰源的“面密度”（buildings/km^2 同量纲：sources/km^2），可由 scale 缩放
        base_lambda_per_km2 = {
            0: 1854.0,  # WiFi 2.4
            1: 1854.0,  # WiFi 5.8
            2: 6.0,   # 4G
            3: 8.0,   # 5G
            4: 1.0,   # GNSS jammer
            5: 6.0,   # industrial
            6: 0.5,   # satellite ground
            7: 0.0,   # UE UL（默认不启用；可用 bg_lambda_km2_7 显式打开）
        }

        # 每类的高度/位置混合（地面/屋顶/室内楼层）
        building_obs = self._compute_building_observables()
        self._building_observables_cache = building_obs
        wifi_density = self._estimate_wifi_ap_density(building_obs)
        self._wifi_density_cache = wifi_density if wifi_density is not None else {}
        if wifi_density:
            if 'lambda_24_km2' in wifi_density:
                base_lambda_per_km2[0] = float(max(0.0, wifi_density['lambda_24_km2']))
            if 'lambda_58_km2' in wifi_density:
                base_lambda_per_km2[1] = float(max(0.0, wifi_density['lambda_58_km2']))

        loc_mix = {
            0: {'ground': 0.10, 'rooftop': 0.10, 'indoor': 0.80},
            1: {'ground': 0.10, 'rooftop': 0.10, 'indoor': 0.80},
            2: {'ground': 0.05, 'rooftop': 0.90, 'indoor': 0.05},
            3: {'ground': 0.10, 'rooftop': 0.85, 'indoor': 0.05},
            4: {'ground': 0.90, 'rooftop': 0.05, 'indoor': 0.05},
            5: {'ground': 0.30, 'rooftop': 0.20, 'indoor': 0.50},
            6: {'ground': 0.95, 'rooftop': 0.05, 'indoor': 0.00},
            7: {'ground': 0.30, 'rooftop': 0.05, 'indoor': 0.65},
        }

        # 允许用户用 user_params 覆盖某些密度（不会在交互里强制询问）
        for type_id in list(base_lambda_per_km2.keys()):
            key = f'bg_lambda_km2_{type_id}'
            if key in self.user_params:
                base_lambda_per_km2[type_id] = float(self.user_params[key])

        sources = []
        for type_id, lam in base_lambda_per_km2.items():
            density_mode = str(getattr(self, 'background_interference_density_mode', '2d')).strip().lower()
            scale = float(getattr(self, 'background_interference_density_scale', 1.0))
            if density_mode == '3d':
                # ρ（sources/m^3） -> λ（sources/km^2）：λ = ρ * Z * 1e6
                z_box = max(float(getattr(self, 'background_interference_box_height_m', self.building_height_max)), 1.0)
                rho_key = f'bg_rho_m3_{int(type_id)}'
                if rho_key in self.user_params:
                    rho_m3 = max(0.0, float(self.user_params[rho_key]))
                    lam_eff = float(rho_m3 * z_box * 1e6 * scale)
                else:
                    lam_eff = float(max(0.0, lam) * scale)
            else:
                lam_eff = float(max(0.0, lam) * scale)
            if lam_eff <= 0.0:
                continue
            n = int(np.random.poisson(lam_eff * area_km2))
            if n <= 0:
                continue

            cfg = self.interference_config.get_config(int(type_id))
            min_p, max_p = cfg['power_range']

            for _ in range(n):
                mix = loc_mix.get(int(type_id), {'ground': 1.0, 'rooftop': 0.0, 'indoor': 0.0})
                allowed = set(self.interference_config.get_allowed_semantics(int(type_id)))
                u = float(np.random.random())
                if not self.buildings:
                    placement = 'ground'
                else:
                    candidates = []
                    weights = []
                    if 'indoor' in allowed:
                        candidates.append('indoor'); weights.append(float(mix.get('indoor', 0.0)))
                    if any(s in allowed for s in ('roof', 'tower', 'pole')):
                        candidates.append('rooftop'); weights.append(float(mix.get('rooftop', 0.0)))
                    if any(s in allowed for s in ('street', 'sidewalk', 'ground')):
                        candidates.append('ground'); weights.append(float(mix.get('ground', 0.0)))
                    if not candidates:
                        candidates = ['ground', 'rooftop', 'indoor']
                        weights = [float(mix.get('ground', 0.0)), float(mix.get('rooftop', 0.0)), float(mix.get('indoor', 0.0))]
                    ws = float(sum(weights))
                    if ws <= 1e-12:
                        placement = candidates[int(u * len(candidates)) % len(candidates)]
                    else:
                        r = u * ws
                        acc = 0.0
                        placement = candidates[-1]
                        for c, w in zip(candidates, weights):
                            acc += float(w)
                            if r <= acc:
                                placement = c
                                break

                power = float(np.random.uniform(min_p, max_p))

                if placement == 'ground':
                    # 先按原公式采样，再做“场景一致性”重采样/投影
                    x = float(np.random.uniform(0.0, self.area_size))
                    y = float(np.random.uniform(0.0, self.area_size))
                    if self._building_idx_containing_xy(x, y) is not None:
                        # 地面源不允许落进建筑平面，重采样
                        for _try in range(int(self.scene_consistency_max_attempts)):
                            x = float(np.random.uniform(0.0, self.area_size))
                            y = float(np.random.uniform(0.0, self.area_size))
                            if self._building_idx_containing_xy(x, y) is None:
                                break
                    semantic = self._classify_semantic(x, y, 0.0, building=None, is_indoor=False)
                    if allowed and semantic not in allowed and 'ground' not in allowed:
                        # 目标优先级：street > sidewalk > ground
                        target = 'street' if 'street' in allowed else ('sidewalk' if 'sidewalk' in allowed else 'ground')
                        if str(self.scene_consistency_mode) == 'project':
                            x, y = self._project_xy_to_semantic(x, y, target)
                            semantic = self._classify_semantic(x, y, 0.0, building=None, is_indoor=False)
                        else:
                            ok = False
                            for _try in range(int(self.scene_consistency_max_attempts)):
                                x = float(np.random.uniform(0.0, self.area_size))
                                y = float(np.random.uniform(0.0, self.area_size))
                                if self._building_idx_containing_xy(x, y) is not None:
                                    continue
                                semantic = self._classify_semantic(x, y, 0.0, building=None, is_indoor=False)
                                if semantic in allowed:
                                    ok = True
                                    break
                            if not ok:
                                x, y = self._project_xy_to_semantic(x, y, target)
                                semantic = self._classify_semantic(x, y, 0.0, building=None, is_indoor=False)
                    src = InterferenceSource(x, y, 0.0, int(type_id), power,
                                             building=None, is_ground=True,
                                             building_idx=None, is_indoor=False, skip_height_clip=True)
                    src.semantic = semantic
                    idx_local = int(len(sources))
                    self._calibrate_source_profile_and_power(idx_local, src, float(power))
                    src.activity = float(self._activity_factor_for_source(idx_local, src))
                    sources.append(src)
                    continue

                # rooftop / indoor：随机选择建筑
                b_idx = int(np.random.randint(0, len(self.buildings)))
                b = self.buildings[b_idx]
                x = float(np.random.uniform(b.x - b.width / 2, b.x + b.width / 2))
                y = float(np.random.uniform(b.y - b.length / 2, b.y + b.length / 2))

                if placement == 'rooftop':
                    rooftop_offset_m = float(cfg.get('rooftop_offset_m', 2.0))
                    h = float(b.height + rooftop_offset_m)
                    src = InterferenceSource(x, y, h, int(type_id), power,
                                             building=b, is_ground=False,
                                             building_idx=b_idx, is_indoor=False, skip_height_clip=True)
                    src.semantic = 'roof'
                    idx_local = int(len(sources))
                    self._calibrate_source_profile_and_power(idx_local, src, float(power))
                    src.activity = float(self._activity_factor_for_source(idx_local, src))
                    sources.append(src)
                    continue

                # indoor：按楼层离散化（楼层高 floor_height_m），高度 = (floor-1)*H + indoor_device_height_m
                floor_h = max(float(self.floor_height_m), 0.5)
                floors_total = max(1, int(math.floor(float(b.height) / floor_h)))
                floor = int(np.random.randint(1, floors_total + 1))
                h = float((floor - 1) * floor_h + float(self.indoor_device_height_m))
                indoor_d = float(self.p1238_d_in_m)
                src = InterferenceSource(x, y, h, int(type_id), power,
                                         building=b, is_ground=False,
                                         building_idx=b_idx, is_indoor=True,
                                         indoor_floor=floor, indoor_floors_total=floors_total,
                                         indoor_distance_m=indoor_d,
                                         skip_height_clip=True)
                src.semantic = 'indoor'
                idx_local = int(len(sources))
                self._calibrate_source_profile_and_power(idx_local, src, float(power))
                src.activity = float(self._activity_factor_for_source(idx_local, src))
                sources.append(src)

        # 限制背景源总数，避免极端参数导致性能问题（保持确定性：按功率排序取前 max）
        if len(sources) > int(self.background_interference_max_sources):
            sources.sort(key=lambda s: float(s.power), reverse=True)
            sources = sources[:int(self.background_interference_max_sources)]

        self.background_interference_sources = sources

        # 可视化只取少量（同样按功率排序取前 max_sources_to_plot）
        viz_n = max(0, int(self.max_sources_to_plot))
        if viz_n > 0:
            viz_sources = list(sources)
            viz_sources.sort(key=lambda s: float(s.power), reverse=True)
            self.background_interference_sources_viz = viz_sources[:viz_n]
        else:
            self.background_interference_sources_viz = []

    def _build_building_spatial_index(self):
        """构建建筑物空间索引（网格），加速点落楼与线段遮挡判断。"""
        self._building_grid = {}
        self._building_grid_cell_size = float(max(30.0, min(150.0, self.area_size / 20.0)))

        if not getattr(self, 'buildings', None):
            self._building_centers_xy = np.zeros((0, 2), dtype=float)
            return

        centers = []
        cs = self._building_grid_cell_size
        for idx, b in enumerate(self.buildings):
            centers.append([float(b.x), float(b.y)])

            xmin = float(b.x - b.width / 2)
            xmax = float(b.x + b.width / 2)
            ymin = float(b.y - b.length / 2)
            ymax = float(b.y + b.length / 2)

            cx0 = int(math.floor(xmin / cs))
            cx1 = int(math.floor(xmax / cs))
            cy0 = int(math.floor(ymin / cs))
            cy1 = int(math.floor(ymax / cs))

            for cx in range(cx0, cx1 + 1):
                for cy in range(cy0, cy1 + 1):
                    self._building_grid.setdefault((cx, cy), []).append(idx)

        self._building_centers_xy = np.asarray(centers, dtype=float)

    def _generate_semantic_layers(self):
        """
        生成“场景标签层”（不改变原生成公式，只用于场景一致性约束/投影）：
        - street：道路车行带
        - sidewalk：人行道
        - ground：其余地面（街区/空地）
        - roof：屋顶
        - indoor：室内楼层
        - facade：建筑外立面附近（近似）
        - pole：路侧杆塔点（近似）
        """
        spacing = float(getattr(self, 'semantic_road_spacing_m', 0.0))
        if spacing <= 1e-6:
            spacing = float(getattr(self, 'building_geom_S_m', 0.0))
            if spacing <= 1e-6:
                spacing = float(max(80.0, self.area_size / 5.0))
            spacing = float(np.clip(spacing * 2.0, 80.0, 300.0))

        road_w = float(max(getattr(self, 'semantic_road_width_m', 20.0), 5.0))
        sidewalk_w = float(max(getattr(self, 'semantic_sidewalk_width_m', 6.0), 0.0))
        facade_band = float(max(getattr(self, 'semantic_facade_band_m', 2.0), 0.0))
        pole_snap = float(max(getattr(self, 'semantic_pole_snap_m', 3.0), 0.0))

        self.semantic_layers = {
            'road_spacing_m': spacing,
            'road_width_m': road_w,
            'sidewalk_width_m': sidewalk_w,
            'facade_band_m': facade_band,
            'pole_snap_m': pole_snap,
        }

    def _road_distance_components(self, x: float, y: float) -> Tuple[float, float]:
        """到最近竖向/横向道路中心线的距离（m）。"""
        spacing = float(getattr(self, 'semantic_layers', {}).get('road_spacing_m', 100.0))
        if spacing <= 1e-9:
            return float('inf'), float('inf')
        xm = float(x) % spacing
        ym = float(y) % spacing
        dv = min(xm, spacing - xm)
        dh = min(ym, spacing - ym)
        return float(dv), float(dh)

    def _classify_semantic(self,
                           x: float,
                           y: float,
                           z: float,
                           building: Optional[Building] = None,
                           is_indoor: bool = False) -> str:
        if is_indoor:
            return 'indoor'
        if building is not None and z >= float(building.height) - 1e-6:
            return 'roof'
        # 关键修复：若已经“绑定到建筑”，但高度在屋顶以下且未标记室内，
        # 则不应被归类为 street/sidewalk（会导致“干扰源漂浮在空中”的错觉/不一致）。
        # 统一把这类点视作建筑相关的“facade”（外立面设备）语义，后续再由 Γ(type) 约束转为 indoor/roof/pole 等合法集合。
        if building is not None and z > 0.5:
            return 'facade'

        # 空域：离地且不在建筑内
        if z > 1.0 and building is None:
            return 'air'

        dv, dh = self._road_distance_components(x, y)
        road_half = float(getattr(self, 'semantic_layers', {}).get('road_width_m', 20.0)) / 2.0
        sidewalk_w = float(getattr(self, 'semantic_layers', {}).get('sidewalk_width_m', 6.0))
        d_road = min(dv, dh)
        if d_road <= road_half:
            return 'street'
        if d_road <= road_half + sidewalk_w:
            pole_snap = float(getattr(self, 'semantic_layers', {}).get('pole_snap_m', 3.0))
            if dv <= road_half + sidewalk_w and dh <= road_half + sidewalk_w and min(dv, dh) <= pole_snap:
                return 'pole'
            return 'sidewalk'
        return 'ground'

    def _project_xy_to_semantic(self, x: float, y: float, target: str) -> Tuple[float, float]:
        """将 (x,y) 投影到最近的目标语义集合（简化几何投影）。"""
        x = float(np.clip(x, 0.0, self.area_size))
        y = float(np.clip(y, 0.0, self.area_size))
        spacing = float(getattr(self, 'semantic_layers', {}).get('road_spacing_m', 100.0))
        if spacing <= 1e-9:
            return x, y

        road_half = float(getattr(self, 'semantic_layers', {}).get('road_width_m', 20.0)) / 2.0
        sidewalk_w = float(getattr(self, 'semantic_layers', {}).get('sidewalk_width_m', 6.0))

        kx = round(x / spacing)
        ky = round(y / spacing)
        cx = float(kx * spacing)
        cy = float(ky * spacing)
        dv, dh = self._road_distance_components(x, y)

        if target == 'street':
            if dv <= dh:
                return float(np.clip(cx, 0.0, self.area_size)), y
            return x, float(np.clip(cy, 0.0, self.area_size))

        if target in ('sidewalk', 'pole'):
            offset = road_half + sidewalk_w / 2.0
            if target == 'pole':
                x0 = float(np.clip(cx, 0.0, self.area_size))
                y0 = float(np.clip(cy, 0.0, self.area_size))
                return float(np.clip(x0 + offset, 0.0, self.area_size)), float(np.clip(y0 + offset, 0.0, self.area_size))

            if dv <= dh:
                sign = 1.0 if (x - cx) >= 0.0 else -1.0
                return float(np.clip(cx + sign * offset, 0.0, self.area_size)), y
            sign = 1.0 if (y - cy) >= 0.0 else -1.0
            return x, float(np.clip(cy + sign * offset, 0.0, self.area_size))

        return x, y

    def _building_idx_containing_xy(self, x: float, y: float) -> Optional[int]:
        """若 (x,y) 落在某建筑平面投影内，返回其索引，否则返回 None。"""
        if not getattr(self, 'buildings', None):
            return None
        cs = float(getattr(self, '_building_grid_cell_size', 100.0))
        cx = int(math.floor(float(x) / cs))
        cy = int(math.floor(float(y) / cs))
        for idx in self._building_grid.get((cx, cy), []):
            b = self.buildings[int(idx)]
            if b.contains_point(float(x), float(y)):
                return int(idx)
        return None

    def _find_or_snap_building(self, x: float, y: float) -> Tuple[Optional[int], Optional[Building]]:
        """根据 (x,y) 找到包含该点的建筑；若没有，返回最近建筑（用于 location_type=建筑物）。"""
        if not self.buildings:
            return None, None

        cs = float(getattr(self, '_building_grid_cell_size', 100.0))
        cx = int(math.floor(float(x) / cs))
        cy = int(math.floor(float(y) / cs))
        for idx in self._building_grid.get((cx, cy), []):
            b = self.buildings[idx]
            if b.contains_point(float(x), float(y)):
                return int(idx), b

        # 未落在任何建筑 footprint 内：确定性地“吸附”到最近建筑
        centers = getattr(self, '_building_centers_xy', None)
        if centers is None or len(centers) == 0:
            return None, None
        dx = centers[:, 0] - float(x)
        dy = centers[:, 1] - float(y)
        nearest_idx = int(np.argmin(dx * dx + dy * dy))
        return nearest_idx, self.buildings[nearest_idx]

    def _candidate_building_indices_for_segment(self, p0: np.ndarray, p1: np.ndarray) -> List[int]:
        """根据线段 (p0,p1) 的 XY 包围盒，返回可能相交的建筑索引集合（去重）。"""
        if not self.buildings:
            return []
        cs = float(getattr(self, '_building_grid_cell_size', 100.0))
        x0 = float(p0[0]); y0 = float(p0[1])
        x1 = float(p1[0]); y1 = float(p1[1])
        xmin = min(x0, x1); xmax = max(x0, x1)
        ymin = min(y0, y1); ymax = max(y0, y1)
        cx0 = int(math.floor(xmin / cs))
        cx1 = int(math.floor(xmax / cs))
        cy0 = int(math.floor(ymin / cs))
        cy1 = int(math.floor(ymax / cs))

        idx_set = set()
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                for idx in self._building_grid.get((cx, cy), []):
                    idx_set.add(int(idx))
        return list(idx_set)

    def _is_segment_blocked_by_buildings(self,
                                         p0: np.ndarray,
                                         p1: np.ndarray,
                                         ignore_building_idx: Optional[int] = None) -> bool:
        """判断线段 p0->p1 是否被任何建筑遮挡（可忽略干扰源所在建筑）。"""
        candidates = self._candidate_building_indices_for_segment(p0, p1)
        if not candidates:
            return False
        for idx in candidates:
            if ignore_building_idx is not None and int(idx) == int(ignore_building_idx):
                continue
            if self.buildings[idx].intersects_segment_3d(p0, p1):
                return True
        return False

    def _generate_station_positions(self):
        """生成地面站位置"""
        self.station_positions = []

        # 在地面均匀分布地面站
        for i in range(self.num_stations):
            if self.num_stations == 1:
                x = self.area_size / 2
                y = self.area_size / 2
            elif self.num_stations == 2:
                x = self.area_size * (i + 1) / 3
                y = self.area_size / 2
            elif self.num_stations == 3:
                angle = 2 * np.pi * i / 3
                x = self.area_size / 2 + self.area_size / 3 * np.cos(angle)
                y = self.area_size / 2 + self.area_size / 3 * np.sin(angle)
            else:
                # 网格分布
                rows = int(np.sqrt(self.num_stations))
                cols = int(np.ceil(self.num_stations / rows))

                row = i // cols
                col = i % cols

                x = (col + 0.5) * self.area_size / cols
                y = (row + 0.5) * self.area_size / rows

            # 避免站点落在建筑内部（启用“几何 LoS”时会导致链路被必然遮挡）
            x = float(x)
            y = float(y)
            for _ in range(30):
                if not any(b.contains_point(x, y) for b in getattr(self, 'buildings', []) or []):
                    break
                x = float(np.random.uniform(0.0, self.area_size))
                y = float(np.random.uniform(0.0, self.area_size))

            self.station_positions.append([x, y, 2.0])  # 地面站高度2m

        self.station_positions = np.array(self.station_positions)

    def _generate_drone_xy_positions(self):
        """生成无人机水平位置"""
        xy = np.random.rand(self.num_drones, 2) * self.area_size
        # 同样避免无人机初始水平位置落在建筑内部（几何 LoS 下会产生不合理的“穿楼起飞”链路）
        for i in range(int(self.num_drones)):
            x = float(xy[i, 0])
            y = float(xy[i, 1])
            for _ in range(30):
                if not any(b.contains_point(x, y) for b in getattr(self, 'buildings', []) or []):
                    break
                x = float(np.random.uniform(0.0, self.area_size))
                y = float(np.random.uniform(0.0, self.area_size))
            xy[i, 0] = x
            xy[i, 1] = y
        self.drone_xy_positions = xy

    def _project_drone_xy_outside_buildings(self, xy: np.ndarray) -> np.ndarray:
        """将无人机水平位置从建筑内部投影到外部（确定性）。"""
        if xy is None:
            return xy
        buildings = getattr(self, 'buildings', []) or []
        if not buildings:
            return np.clip(np.asarray(xy, dtype=float), 0.0, self.area_size)

        out = np.array(xy, dtype=float, copy=True)
        eps = max(float(self.drone_xy_snap_eps_m), 1e-3)
        max_iter = max(int(self.drone_xy_project_max_iter), 1)

        for i in range(out.shape[0]):
            x = float(out[i, 0])
            y = float(out[i, 1])
            for _ in range(max_iter):
                best_move = None
                for b in buildings:
                    if not b.contains_point(x, y):
                        continue
                    xmin = float(b.x - b.width / 2)
                    xmax = float(b.x + b.width / 2)
                    ymin = float(b.y - b.length / 2)
                    ymax = float(b.y + b.length / 2)
                    candidates = [
                        (abs(x - xmin), xmin - eps, y),
                        (abs(xmax - x), xmax + eps, y),
                        (abs(y - ymin), x, ymin - eps),
                        (abs(ymax - y), x, ymax + eps),
                    ]
                    local = min(candidates, key=lambda item: item[0])
                    if best_move is None or local[0] < best_move[0]:
                        best_move = local
                if best_move is None:
                    break
                x = float(best_move[1])
                y = float(best_move[2])
            x = float(np.clip(x, 0.0, self.area_size))
            y = float(np.clip(y, 0.0, self.area_size))
            out[i, 0] = x
            out[i, 1] = y

        return out

    @staticmethod
    def _is_segment_blocked_by_buildings_naive(p0: np.ndarray,
                                               p1: np.ndarray,
                                               buildings: List[Building],
                                               ignore_building: Optional[Building] = None) -> bool:
        """无遮挡加速索引时的兜底：逐栋检查线段是否与建筑体相交。"""
        if not buildings:
            return False
        for b in buildings:
            if ignore_building is not None and b is ignore_building:
                continue
            if b.intersects_segment_3d(p0, p1):
                return True
        return False

    def check_dimensions(self):
        """检查维度是否正确"""
        print("\n维度检查:")
        print(f"无人机数量: {self.num_drones}")
        print(f"干扰源数量: {self.num_interference}")
        print(f"Geatpy维度: {self.Dim}")

        # 验证各个部分的维度
        total_vars = (self.num_drones * 4 +  # 无人机高度、速度、水平位置(x,y)
                      self.num_interference * 5)  # 干扰源类型、功率、位置类型、位置(x,y)
        print(f"计算总维度: {total_vars}")

        if self.Dim != total_vars:
            print(f"错误: 维度不匹配! Geatpy维度: {self.Dim}, 计算维度: {total_vars}")
            return False
        else:
            print("维度正确!")
            return True

    def generate_scenario(self, x: np.ndarray) -> Dict:
        """根据染色体生成场景"""
        idx = 0

        # 1. 无人机高度
        drone_heights = x[idx:idx + self.num_drones]
        idx += self.num_drones

        # 2. 无人机速度
        drone_speeds = x[idx:idx + self.num_drones]
        idx += self.num_drones

        # 3. 无人机水平位置
        drone_xy = x[idx:idx + self.num_drones * 2].reshape(self.num_drones, 2)
        idx += self.num_drones * 2
        drone_xy = np.clip(drone_xy, 0.0, self.area_size)
        if self.drone_xy_avoid_buildings:
            drone_xy = self._project_drone_xy_outside_buildings(drone_xy)

        # 无人机完整位置
        drone_positions = np.zeros((self.num_drones, 3))
        drone_positions[:, :2] = drone_xy
        drone_positions[:, 2] = drone_heights

        # 4. 干扰源类型
        interference_types = [int(t) for t in x[idx:idx + self.num_interference]]
        idx += self.num_interference

        # 5. 干扰源功率
        interference_powers = x[idx:idx + self.num_interference]
        idx += self.num_interference

        # 6. 干扰源位置类型 (0=地面, 1=建筑物)
        interference_location_types = [int(t) for t in x[idx:idx + self.num_interference]]
        idx += self.num_interference

        # 7. 干扰源位置
        interference_positions = x[idx:idx + self.num_interference * 2]

        # 生成“受控干扰源”（由染色体控制）
        controlled_sources = []
        for i in range(self.num_interference):
            type_id = interference_types[i]
            power = interference_powers[i]
            location_type = interference_location_types[i]

            # 获取位置坐标
            pos_x = interference_positions[i * 2]
            pos_y = interference_positions[i * 2 + 1]

            # 根据位置类型确定高度和建筑物
            if location_type == 1:  # 建筑物上
                # 先裁剪到区域内，再根据 (x,y) 找到/吸附到最近建筑
                pos_x = float(np.clip(pos_x, 0.0, self.area_size))
                pos_y = float(np.clip(pos_y, 0.0, self.area_size))
                building_idx, building = self._find_or_snap_building(pos_x, pos_y)
                if building is None:
                    # 没有建筑物可用，退化为地面干扰源
                    source = InterferenceSource(pos_x, pos_y, 0.0, type_id, power, None, is_ground=True, building_idx=None)
                else:
                    # 确保位置在建筑物范围内
                    pos_x = float(np.clip(pos_x, building.x - building.width / 2, building.x + building.width / 2))
                    pos_y = float(np.clip(pos_y, building.y - building.length / 2, building.y + building.length / 2))

                    # 根据干扰源类型确定高度（确定性选择，降低 GA 评估噪声）
                    config = self.interference_config.get_config(type_id)
                    suitable_heights = building.get_suitable_interference_heights(config['name'])
                    height = self._select_deterministic_height(i, type_id, power, pos_x, pos_y, suitable_heights)

                    source = InterferenceSource(pos_x, pos_y, height, type_id, power, building,
                                                is_ground=False, building_idx=building_idx)
            else:  # 地面干扰源
                pos_x = float(np.clip(pos_x, 0.0, self.area_size))
                pos_y = float(np.clip(pos_y, 0.0, self.area_size))
                height = 0.0
                source = InterferenceSource(pos_x, pos_y, height, type_id, power, None, is_ground=True, building_idx=None)

            source = self._apply_scene_constraints_to_controlled_source(int(i), source)
            # 基于“可追溯 profile”对功率做标定（不改变染色体维度，只改变解释/映射）
            self._calibrate_source_profile_and_power(int(i), source, float(power))
            if not hasattr(source, 'semantic'):
                source.semantic = self._classify_semantic(float(source.x), float(source.y), float(source.height),
                                                          building=source.building, is_indoor=bool(source.is_indoor))
            controlled_sources.append(source)

        # 背景干扰源（PPP 生成，多类型多高度）
        all_sources = list(controlled_sources)
        if self.background_interference_enabled and getattr(self, 'background_interference_sources', None):
            all_sources.extend(self.background_interference_sources)

        # 可视化子集：受控源 + 背景子集（避免画面拥挤）
        viz_sources = list(controlled_sources)
        if self.background_interference_enabled and getattr(self, 'background_interference_sources_viz', None):
            viz_sources.extend(self.background_interference_sources_viz)

        scenario = {
            'drone_positions': drone_positions,
            'drone_speeds': drone_speeds,
            'station_positions': self.station_positions,
            'interference_sources': all_sources,
            'interference_sources_viz': viz_sources,
            'controlled_interference_sources': controlled_sources,
            'buildings': self.buildings,
            'semantic_layers': getattr(self, 'semantic_layers', {}),
            'area_size': self.area_size,
            'building_density': self.building_density,
            'drone_speed_max': self.drone_speed_max,
            'drone_height_max': self.drone_height_max,
            'building_height_max': self.building_height_max
        }

        # 预计算：每架无人机处的总干扰功率（mW），避免在每条 A2G 链路上重复计算
        scenario['floor_height_m'] = float(getattr(self, 'floor_height_m', 3.0))
        if getattr(self, '_building_observables_cache', None) is not None:
            scenario['building_observables'] = self._building_observables_cache
        if getattr(self, '_wifi_density_cache', None) is not None:
            scenario['wifi_density'] = self._wifi_density_cache

        scenario['interference_power_mw_per_drone'] = self._precompute_interference_power_mw_per_drone(
            drone_positions=scenario['drone_positions'],
            interference_sources=scenario.get('interference_sources', []),
            buildings=scenario.get('buildings', [])
        )

        return scenario

    @staticmethod
    def _stable_unit_float(*values: float) -> float:
        """
        基于输入数值生成稳定的 [0,1) “伪随机”数（不依赖全局 RNG 状态）。
        用于让同一染色体在重复评估时得到确定性场景，从而降低 GA 目标噪声。
        """
        acc = 0.0
        for i, v in enumerate(values):
            acc += float(v) * (12.9898 + 17.0 * i)
        s = math.sin(acc) * 43758.5453
        return float(s - math.floor(s))

    def _activity_factor_for_source(self, source_index: int, source: 'InterferenceSource') -> float:
        """
        返回“平均占空/活动因子”∈[0,1]，用于把峰值/瞬时 EIRP 折算为平均干扰强度。
        - Wi‑Fi：traffic duty + beacon 周期性占空（可通过 user_params 覆盖）
        - 5G小基站：活动因子 70%（图片大纲锚点）
        - 其他：默认 1
        """
        type_id = int(getattr(source, 'type_id', -1))
        cfg = self.interference_config.get_config(type_id)
        type_key = self.interference_config.type_mapping.get(type_id, None)
        cal = (cfg.get('calibration', None) or {})

        if type_key in ('wifi_2_4g', 'wifi_5_8g'):
            beacon_period_s = float(cal.get('wifi_beacon_period_s', 0.1024))
            beacon_dur_ms = float(self.user_params.get(
                'wifi_beacon_duration_ms',
                cal.get('wifi_beacon_duration_ms_default', 0.5)
            ))
            traffic_dc = float(self.user_params.get(
                'wifi_traffic_duty_cycle',
                cal.get('wifi_traffic_duty_cycle_default', 0.15)
            ))
            beacon_dc = float(max(0.0, beacon_dur_ms) / 1000.0 / max(beacon_period_s, 1e-9))
            dc = float(np.clip(max(0.0, traffic_dc) + beacon_dc, 0.0, 1.0))
            return dc

        if type_key in ('cellular_4g', 'cellular_5g'):
            alpha_avg = float(self._traffic_profile_alpha_avg())
            cap = float(getattr(self, 'bs_capacity_max', 100.0))
            users = float(getattr(self, 'bs_users_per_station', 0.0))
            load = 0.0 if cap <= 0.0 else float(np.clip(users / cap, 0.0, 1.0))
            load = float(np.clip(load * alpha_avg, 0.0, 1.0))
            if load > 0.0:
                load = max(load, float(getattr(self, 'bs_load_floor', 0.0)))
            if type_key == 'cellular_5g':
                base_act = float(np.clip(float(cal.get('activity_factor', 1.0)), 0.0, 1.0))
                return float(np.clip(load * base_act, 0.0, 1.0))
            return float(np.clip(load, 0.0, 1.0))

        return 1.0

    def _ue_ul_fpc_power_dbm(self,
                            source_index: int,
                            source: 'InterferenceSource',
                            raw_power_dbm: float,
                            fpc_cfg: Dict) -> Tuple[float, Dict]:
        """
        蜂窝 UE 上行分数功控（FPC）标定：
        Pu = min(Pmax, P0 + α·PL + 10log10(M) + ΔTF + f(ΔTPC))
        说明：PL 为 UE→服务小区路径损耗（这里用“典型服务距离 + FSPL + 额外损耗”做可标定代理）。
        """
        pmax_dbm = float(fpc_cfg.get('pmax_dbm', 23.0))
        p0_candidates = list(fpc_cfg.get('p0_candidates_dbm', [-88.0]))
        if not p0_candidates:
            p0_candidates = [-88.0]

        m_candidates = list(fpc_cfg.get('m_rb_candidates', [1]))
        if not m_candidates:
            m_candidates = [1]

        # 用 raw_power_dbm “编码”选择 P0 档位（保持 GA 仍能探索该维度）
        u_p0 = float(np.clip(float(raw_power_dbm) / 70.0, 0.0, 1.0 - 1e-12))
        p0_dbm = float(p0_candidates[int(u_p0 * len(p0_candidates))])

        u_m = self._stable_unit_float(source_index, float(source.x), float(source.y), float(source.height), 9.17)
        m_rb = int(m_candidates[int(np.clip(u_m, 0.0, 1.0 - 1e-12) * len(m_candidates))])
        m_rb = max(1, int(m_rb))

        alpha_switch_h = float(fpc_cfg.get('alpha_switch_height_m', 100.0))
        alpha = float(fpc_cfg.get('alpha_ground_or_low_uav', 0.8)) if float(source.height) <= alpha_switch_h else float(
            fpc_cfg.get('alpha_high_uav', 0.7)
        )

        dmin, dmax = fpc_cfg.get('serving_distance_m_range', (30.0, 500.0))
        dmin = float(max(dmin, 1.0))
        dmax = float(max(dmax, dmin))
        u_d = self._stable_unit_float(source_index, float(source.x), float(source.y), 4.2)
        # log-uniform in [dmin, dmax]
        d_serv = float(dmin * ((dmax / dmin) ** float(np.clip(u_d, 0.0, 1.0))))

        pl_db = float(self.calculate_fspl_db_at_freq(d_serv, float(source.frequency)))
        pl_extra_db = float(self.user_params.get('ue_ul_pl_extra_db', fpc_cfg.get('pl_extra_db_default', 20.0)))
        pl_db = float(pl_db + max(0.0, pl_extra_db))

        delta_tf_db = float(fpc_cfg.get('delta_tf_db', 0.0))
        f_delta_tpc_db = float(fpc_cfg.get('f_delta_tpc_db', 0.0))
        tx_dbm = float(p0_dbm + alpha * pl_db + 10.0 * np.log10(float(m_rb)) + delta_tf_db + f_delta_tpc_db)
        tx_dbm = float(min(pmax_dbm, tx_dbm))
        tx_dbm = float(max(-50.0, tx_dbm))

        meta = {
            'fpc': {
                'pmax_dbm': pmax_dbm,
                'p0_dbm': p0_dbm,
                'alpha': alpha,
                'm_rb': m_rb,
                'pl_serv_db': pl_db,
                'pl_serv_distance_m': d_serv,
                'pl_extra_db': pl_extra_db,
                'delta_tf_db': delta_tf_db,
                'f_delta_tpc_db': f_delta_tpc_db
            }
        }
        return tx_dbm, meta

    def _calibrate_source_profile_and_power(self,
                                           source_index: int,
                                           source: 'InterferenceSource',
                                           raw_power_dbm: float) -> None:
        """
        把“宽泛功率范围”映射为可追溯的标定 profile：
        - 5G：femto/pico/micro 锚点 + 小偏移
        - Wi‑Fi：按 band 的最小功率门槛裁剪
        - UE UL：按 FPC 公式计算 TxPower
        其余类型保持原功率（仅做裁剪/记录）。
        """
        type_id = int(getattr(source, 'type_id', -1))
        type_key = self.interference_config.type_mapping.get(type_id, 'unknown')
        cfg = self.interference_config.get_config(type_id)
        cal = (cfg.get('calibration', None) or {})

        source.raw_power_dbm = float(raw_power_dbm)
        source.profile_key = str(type_key)
        profile_meta: Dict = {
            'type_key': str(type_key),
            'sources': dict(cal.get('sources', {}) or {}),
            'power_semantics': str(cfg.get('power_semantics', 'eirp_equivalent_dbm')),
        }

        if type_key == 'cellular_5g':
            anchors = dict(cal.get('eirp_anchors_dbm', {}) or {})
            if anchors:
                p_in = float(source.power)
                anchor_name, anchor_dbm = min(anchors.items(), key=lambda kv: abs(p_in - float(kv[1])))
                anchor_dbm = float(anchor_dbm)
                offset_db = float(np.clip(p_in - anchor_dbm, -3.0, 3.0))
                source.power = float(anchor_dbm + offset_db)
                source.profile_key = f'5g_smallcell_{anchor_name}'
                profile_meta.update({
                    'eirp_anchor_dbm': anchor_dbm,
                    'eirp_offset_db': offset_db,
                })

        elif type_key in ('wifi_2_4g', 'wifi_5_8g'):
            floor_dbm = float(cal.get('eirp_min_dbm', float(cfg.get('power_range', (0.0, 0.0))[0])))
            source.power = float(max(float(source.power), floor_dbm))
            source.profile_key = f'{type_key}_ap'
            profile_meta.update({'eirp_floor_dbm': floor_dbm})

        elif type_key == 'cellular_ue_ul':
            fpc_cfg = dict((cal.get('ul_fpc', None) or {}))
            tx_dbm, ue_meta = self._ue_ul_fpc_power_dbm(source_index, source, raw_power_dbm, fpc_cfg)
            source.power = float(tx_dbm)
            source.profile_key = 'ue_ul_fpc'
            profile_meta.update(ue_meta)

        source.profile_meta = profile_meta

    def _stable_exp_fading_gain(self, drone_idx: int, source_idx: int, source: 'InterferenceSource') -> float:
        """
        可复现的 Rayleigh 小尺度衰落功率增益 h（Exp(1)）。
        - 关闭时返回 1（即用均值，不引入随机性）
        - 开启时仍保持同一场景/染色体重复评估一致，避免 GA 目标噪声
        """
        if not getattr(self, 'interference_fading_enabled', False):
            return 1.0
        u = self._stable_unit_float(
            float(drone_idx) + 1.0,
            float(source_idx) + 1.0,
            float(source.power),
            float(source.x),
            float(source.y),
            float(source.height),
        )
        u = float(np.clip(u, 1e-12, 1.0 - 1e-12))
        return float(-math.log(1.0 - u))

    def _stable_exp_fading_gain_for_link(self, key: Tuple[float, ...]) -> float:
        """可复现的 Rayleigh 功率增益 h~Exp(1)，用于“有用信号”链路。"""
        if not getattr(self, 'signal_fading_enabled', False):
            return 1.0
        u = self._stable_unit_float(*key)
        u = float(np.clip(u, 1e-12, 1.0 - 1e-12))
        return float(-math.log(1.0 - u))

    def _select_deterministic_height(self,
                                    source_index: int,
                                    type_id: int,
                                    power_dbm: float,
                                    pos_x: float,
                                    pos_y: float,
                                    suitable_heights: List[float]) -> float:
        if not suitable_heights:
            return 0.0
        if len(suitable_heights) == 1:
            return float(suitable_heights[0])

        u = self._stable_unit_float(source_index, type_id, power_dbm, pos_x, pos_y)
        idx = int(math.floor(u * len(suitable_heights)))
        idx = max(0, min(idx, len(suitable_heights) - 1))
        return float(suitable_heights[idx])

    def _apply_scene_constraints_to_controlled_source(self, source_index: int, source: InterferenceSource) -> InterferenceSource:
        """
        对 GA 控制的干扰源做“场景一致性”约束（不改你原先的生成公式，只做条件化重采样/投影）：
        - WiFi：强制 indoor
        - 蜂窝：优先 roof/tower（或 pole 近似）
        - GNSS jammer：优先 street/sidewalk
        """
        type_id = int(source.type_id)
        allowed = set(self.interference_config.get_allowed_semantics(type_id))

        # activity：使用“平均占空/活动因子”，避免 Bernoulli 抖动引入目标噪声
        source.activity = float(self._activity_factor_for_source(int(source_index), source))

        semantic = self._classify_semantic(float(source.x), float(source.y), float(source.height),
                                           building=source.building, is_indoor=bool(source.is_indoor))
        # tower 视作 roof 的同义
        allowed_effective = set(allowed)
        if 'tower' in allowed_effective:
            allowed_effective.add('roof')

        if not allowed_effective:
            source.semantic = semantic
            return source
        if semantic in allowed_effective or ('ground' in allowed_effective and semantic == 'ground'):
            source.semantic = semantic
            return source

        mode = str(getattr(self, 'scene_consistency_mode', 'resample')).strip().lower()

        # 1) 优先 indoor
        if 'indoor' in allowed_effective and self.buildings:
            b_idx, b = self._find_or_snap_building(float(source.x), float(source.y))
            if b is not None:
                x = float(np.clip(source.x, b.x - b.width / 2, b.x + b.width / 2))
                y = float(np.clip(source.y, b.y - b.length / 2, b.y + b.length / 2))
                floor_h = max(float(self.floor_height_m), 0.5)
                floors_total = max(1, int(math.floor(float(b.height) / floor_h)))
                u_floor = self._stable_unit_float(source_index, type_id, float(source.power), x, y, 1.414)
                floor = 1 + int(math.floor(u_floor * floors_total))
                floor = max(1, min(floor, floors_total))
                h = float((floor - 1) * floor_h + float(self.indoor_device_height_m))
                new_src = InterferenceSource(x, y, h, type_id, float(source.power),
                                             building=b, is_ground=False, building_idx=b_idx,
                                             is_indoor=True, indoor_floor=floor, indoor_floors_total=floors_total,
                                             indoor_distance_m=float(self.p1238_d_in_m),
                                             skip_height_clip=True)
                new_src.activity = float(self._activity_factor_for_source(int(source_index), new_src))
                new_src.semantic = 'indoor'
                return new_src

        # 2) roof
        if 'roof' in allowed_effective and self.buildings:
            b_idx, b = self._find_or_snap_building(float(source.x), float(source.y))
            if b is not None:
                x = float(np.clip(source.x, b.x - b.width / 2, b.x + b.width / 2))
                y = float(np.clip(source.y, b.y - b.length / 2, b.y + b.length / 2))
                cfg = self.interference_config.get_config(type_id)
                rooftop_offset_m = float(cfg.get('rooftop_offset_m', 2.0))
                h = float(b.height + rooftop_offset_m)
                new_src = InterferenceSource(x, y, h, type_id, float(source.power),
                                             building=b, is_ground=False, building_idx=b_idx,
                                             is_indoor=False, skip_height_clip=True)
                new_src.activity = float(self._activity_factor_for_source(int(source_index), new_src))
                new_src.semantic = 'roof'
                return new_src

        # 3) street/sidewalk：投影/重采样到道路语义
        if any(s in allowed_effective for s in ('street', 'sidewalk', 'pole', 'ground')):
            target = 'street' if 'street' in allowed_effective else ('sidewalk' if 'sidewalk' in allowed_effective else ('pole' if 'pole' in allowed_effective else 'ground'))
            x = float(source.x)
            y = float(source.y)
            if mode == 'project':
                x, y = self._project_xy_to_semantic(x, y, target if target != 'ground' else 'sidewalk')
            else:
                # 确定性重采样：用 stable_unit_float 产生伪随机 (x,y)
                for t in range(int(getattr(self, 'scene_consistency_max_attempts', 12))):
                    u1 = self._stable_unit_float(source_index, type_id, float(source.power), float(x), float(y), 2.0 + t)
                    u2 = self._stable_unit_float(source_index, type_id, float(source.power), float(x), float(y), 3.0 + t)
                    x = float(np.clip(u1 * self.area_size, 0.0, self.area_size))
                    y = float(np.clip(u2 * self.area_size, 0.0, self.area_size))
                    if self._building_idx_containing_xy(x, y) is not None:
                        continue
                    s = self._classify_semantic(x, y, 0.0, building=None, is_indoor=False)
                    if (s in allowed_effective) or ('ground' in allowed_effective and s == 'ground'):
                        break
                if target in ('street', 'sidewalk', 'pole'):
                    x, y = self._project_xy_to_semantic(x, y, target)

            new_sem = self._classify_semantic(x, y, 0.0, building=None, is_indoor=False)
            new_src = InterferenceSource(x, y, 0.0, type_id, float(source.power),
                                         building=None, is_ground=True, building_idx=None,
                                         is_indoor=False, skip_height_clip=True)
            new_src.activity = float(self._activity_factor_for_source(int(source_index), new_src))
            new_src.semantic = new_sem
            return new_src

        source.semantic = semantic
        source.activity = float(self._activity_factor_for_source(int(source_index), source))
        return source

    def _precompute_interference_power_mw_per_drone(self,
                                                    drone_positions: np.ndarray,
                                                    interference_sources: List[InterferenceSource],
                                                    buildings: List[Building]) -> np.ndarray:
        if drone_positions is None or len(drone_positions) == 0:
            return np.zeros((0,), dtype=float)
        if not interference_sources:
            return np.zeros((len(drone_positions),), dtype=float)

        model = str(getattr(self, 'interference_aggregation_model', 'linear')).strip().lower()
        totals = np.zeros((len(drone_positions),), dtype=float)
        if model in ('nakagami', 'nakagami_m', 'moment', 'mm'):
            for drone_idx, drone_pos in enumerate(drone_positions):
                totals[drone_idx] = float(self._nakagami_aggregate_interference_mw(
                    drone_pos=np.asarray(drone_pos, dtype=float),
                    interference_sources=interference_sources,
                    buildings=buildings,
                    drone_idx=int(drone_idx),
                ))
            return totals

        source_cache = []
        for src_idx, source in enumerate(interference_sources):
            p0 = np.array([source.x, source.y, source.height], dtype=float)
            freq_coupling = self._spectral_coupling_factor(
                tx_freq_hz=float(source.frequency),
                rx_freq_hz=float(self.communication_freq),
                rx_bw_hz=float(getattr(self, 'bandwidth_hz', 20e6)),
                interferer_type_id=int(getattr(source, 'type_id', -1)),
            )
            source_cache.append((
                int(src_idx),
                source,
                p0,
                float(freq_coupling),
                float(source.coverage_radius),
                float(source.power),
            ))
        for drone_idx, drone_pos in enumerate(drone_positions):
            total_mw = 0.0
            p1 = np.asarray(drone_pos, dtype=float)
            for src_idx, source, p0, freq_coupling, coverage_radius, source_power_dbm in source_cache:
                distance = float(np.linalg.norm(p0 - p1))
                path_loss_db, _ = self._interference_path_loss_db(
                    source=source,
                    uav_pos=p1,
                    buildings=buildings,
                    p0=p0,
                    distance_m=distance,
                    blocked=None,
                )
                effective_power_dbm = float(source_power_dbm - float(path_loss_db))

                if distance > coverage_radius and coverage_radius > 0.0:
                    excess_ratio = float(distance / coverage_radius)
                    effective_power_dbm -= float(20.0 * np.log10(max(excess_ratio, 1e-6)))

                fading_gain = self._stable_exp_fading_gain(int(drone_idx), int(src_idx), source)
                activity = float(getattr(source, 'activity', 1.0))
                total_mw += self._dbm_to_mw(effective_power_dbm) * float(freq_coupling) * float(fading_gain) * activity
            totals[drone_idx] = float(total_mw)

        return totals

    def calculate_los_probability(self, pos1: np.ndarray, pos2: np.ndarray,
                                 buildings: List[Building]) -> float:
        """
        计算视距（LoS）概率。

        与三张图片对应关系：
        - Sigmoid 近似（推荐，图片 Eq.(5) / 中文(2)）：P_LoS = 1 / (1 + C·exp(-B(θ-C)))
        - ITU 原始乘积（图片 Eq.(4)）：P_LoS = ∏[1-exp(-h_n^2/(2γ^2))], m=floor(r*sqrt(αβ)-1)

        注：两者在论文中是“原始形式 vs 工程化近似”的关系。
        """
        r = float(np.linalg.norm(pos1[:2] - pos2[:2]))  # 水平距离（m）
        h = float(abs(pos1[2] - pos2[2]))               # 高度差（m）

        # 仰角 θ（度）
        if r < 1e-6:
            theta_deg = 90.0
        else:
            theta_deg = float(np.degrees(np.arctan(h / r)))

        model = str(self.atg_model.get('los_model', 'sigmoid')).strip().lower()

        if model in ('sigmoid', 's', 'approx'):
            C = float(self.atg_model.get('los_a', 9.61))
            B = float(self.atg_model.get('los_b', 0.16))
            p_los = 1.0 / (1.0 + C * np.exp(-B * (theta_deg - C)))
            return float(np.clip(p_los, 0.01, 0.99))

        if model in ('sigmoid_fit', 'fit'):
            C = float(self.atg_model.get('los_a', 9.61))
            B = float(self.atg_model.get('los_b', 0.16))
            p_los = 1.0 / (1.0 + C * np.exp(-B * (theta_deg - C)))
            return float(np.clip(p_los, 0.01, 0.99))

        # Height-dependent 原始乘积形式（基于 α、β、γ）
        # 说明：该模型本质是统计模型，与显式建筑列表无直接关系；保留 buildings 参数以兼容调用。
        h_tx = float(max(pos1[2], pos2[2]))
        h_rx = float(min(pos1[2], pos2[2]))

        alpha = float(np.clip(self.itu_city['alpha'], 0.01, 0.99))
        beta_km2 = max(float(self.itu_city['beta_km2']), 1e-6)
        beta_m2 = beta_km2 / 1e6
        gamma = max(float(self.itu_city['gamma_m']), 1e-6)

        # 论文中 r 通常以 km 表示，β 以 buildings/km^2 表示；这里按单位一致性换算：
        # N_b = floor(r_km * sqrt(alpha * beta_km2))
        r_km = r / 1000.0
        nb = int(np.floor(r_km * np.sqrt(alpha * beta_km2)))
        nb = max(0, nb)

        # P(LoS) = Π_{i=0}^{N_b} P_i,  P_i = 1 - exp(-(h_i)^2/(2γ^2))
        prod = 1.0
        denom = float(nb + 1)
        for n in range(nb + 1):
            frac = (n + 0.5) / denom
            h_los = h_tx - frac * (h_tx - h_rx)
            prod *= (1.0 - np.exp(-(h_los ** 2) / (2.0 * (gamma ** 2))))

        return float(np.clip(prod, 0.01, 0.99))

    def calculate_fspl_db_at_freq(self, distance_m: float, freq_hz: float) -> float:
        """自由空间路径损耗（FSPL, dB），支持任意频率。"""
        d = max(float(distance_m), 1e-6)
        f = max(float(freq_hz), 1e-6)
        c = float(self.light_speed)
        return float(20.0 * np.log10(d) + 20.0 * np.log10(f) + 20.0 * np.log10(4.0 * np.pi / c))

    @staticmethod
    def _elevation_angle_deg(tx_pos: np.ndarray, rx_pos: np.ndarray) -> float:
        tx_pos = np.asarray(tx_pos, dtype=float)
        rx_pos = np.asarray(rx_pos, dtype=float)
        r = float(np.linalg.norm(tx_pos[:2] - rx_pos[:2]))
        h = float(abs(tx_pos[2] - rx_pos[2]))
        if r < 1e-9:
            return 90.0
        return float(np.degrees(np.arctan(h / r)))

    def _itu_p1238_floor_loss_db(self, floors: int) -> float:
        floors = int(max(1, floors))
        # 图中给出的可加项：Lf(n)=15+4(n-1)
        return float(15.0 + 4.0 * (floors - 1))

    def _itu_p1238_indoor_loss_db(self, freq_hz: float, distance_m: float, floors: int) -> float:
        if not self.p1238_enabled:
            return 0.0
        f_mhz = max(float(freq_hz) / 1e6, 1e-6)
        d_m = max(float(distance_m), 1.0)
        N = float(self.p1238_N)
        lf = self._itu_p1238_floor_loss_db(floors)
        return float(20.0 * np.log10(f_mhz) + N * np.log10(d_m) + lf - 28.0)

    def _itu_p2109_bel_loss_db(self, freq_hz: float, elevation_deg: float) -> float:
        if not self.p2109_bel_enabled:
            return 0.0
        if None in (self.p2109_bel_r, self.p2109_bel_s, self.p2109_bel_t, self.p2109_bel_u, self.p2109_bel_li_db):
            raise ValueError("启用了 p2109_bel_enabled，但未提供 p2109_bel_r/s/t/u/li_db 参数")
        f_ghz = max(float(freq_hz) / 1e9, 1e-9)
        r = float(self.p2109_bel_r)
        s = float(self.p2109_bel_s)
        t = float(self.p2109_bel_t)
        u = float(self.p2109_bel_u)
        li = float(self.p2109_bel_li_db)
        lh = r + s * np.log10(f_ghz)
        lg = max(float(self.p2109_bel_lg_clip_db), t + u * np.log10(f_ghz))
        le = 0.212 * abs(float(elevation_deg))
        return float(lh + le + lg + li)

    def _interference_path_loss_db(self,
                                   source: InterferenceSource,
                                   uav_pos: np.ndarray,
                                   buildings: List[Building],
                                   p0: Optional[np.ndarray] = None,
                                   distance_m: Optional[float] = None,
                                   blocked: Optional[bool] = None) -> Tuple[float, float]:
        """
        干扰源→无人机路径损耗（dB），按“LoS/NLoS 两态 + 概率加权”的 A2G 形式计算，并返回 (PL_dB, P_LoS)。
        - 室内源：强制 NLoS，并叠加 ITU P.1238 的室内项（可选）与 P.2109 的 BEL（可选）。
        - 室外源：若几何遮挡，则强制 NLoS。
        """
        if p0 is None:
            p0 = np.array([source.x, source.y, source.height], dtype=float)
        p1 = np.asarray(uav_pos, dtype=float)
        if distance_m is None:
            distance = float(np.linalg.norm(p0 - p1))
        else:
            distance = float(distance_m)

        if blocked is None:
            blocked = False
            if buildings:
            # 仅当 buildings 是 self.buildings 时才可安全使用“网格加速”版本（依赖 self._building_grid 与索引一致性）
                if buildings is self.buildings:
                    blocked = self._is_segment_blocked_by_buildings(p0, p1, ignore_building_idx=source.building_idx)
                else:
                    blocked = self._is_segment_blocked_by_buildings_naive(p0, p1, buildings, ignore_building=source.building)

        if source.is_indoor or blocked:
            p_los = 0.0
        else:
            mode = str(getattr(self, 'los_indicator_mode', 'deterministic_if_map')).strip().lower()
            if mode == 'deterministic' or (mode in ('deterministic_if_map', 'hybrid') and len(buildings) > 0):
                p_los = 1.0
            else:
                p_los = float(self.calculate_los_probability(p0, p1, buildings))

        fspl = self.calculate_fspl_db_at_freq(distance, float(source.frequency))
        eta_los = float(self.atg_model['eta_los_db'])
        eta_nlos = float(self.atg_model['eta_nlos_db'])

        pl_los = fspl + eta_los
        pl_nlos = fspl + eta_nlos
        pl = float(p_los * pl_los + (1.0 - p_los) * pl_nlos)

        # 设备类型相关的额外损耗（将“线性域因子”转到 dB 域加法）
        height_factor = float(source._calculate_height_factor(float(uav_pos[2])))
        factor = float(source.path_loss_factor) * float(height_factor)
        pl += float(10.0 * np.log10(max(factor, 1e-12)))

        # 室内额外损耗（可加链）：P.1238 + BEL（可选）
        if source.is_indoor:
            floors = int(source.indoor_floor) if source.indoor_floor is not None else 1
            d_in = float(source.indoor_distance_m) if source.indoor_distance_m is not None else float(self.p1238_d_in_m)
            pl += self._itu_p1238_indoor_loss_db(float(source.frequency), d_in, floors)

            elev = self._elevation_angle_deg(p0, p1)
            pl += self._itu_p2109_bel_loss_db(float(source.frequency), elev)

        return float(pl), float(p_los)

    def calculate_fspl_db(self, distance_m: float) -> float:
        """自由空间路径损耗（FSPL, dB）"""
        d = max(float(distance_m), 1e-6)
        f = float(self.communication_freq)
        c = float(self.light_speed)
        return float(20.0 * np.log10(d) + 20.0 * np.log10(f) + 20.0 * np.log10(4.0 * np.pi / c))

    def calculate_atg_mean_path_loss_db(self, uav_pos: np.ndarray, gt_pos: np.ndarray, buildings: List[Building]) -> float:
        """平均路径损耗 Λ（dB），LoS/NLoS 空间期望"""
        r = float(np.linalg.norm(uav_pos[:2] - gt_pos[:2]))
        h = float(abs(uav_pos[2] - gt_pos[2]))
        d = float(np.sqrt(h ** 2 + r ** 2))

        fspl = self.calculate_fspl_db(d)
        eta_los = float(self.atg_model['eta_los_db'])
        eta_nlos = float(self.atg_model['eta_nlos_db'])

        pl_los = fspl + eta_los
        pl_nlos = fspl + eta_nlos

        mode = str(getattr(self, 'los_indicator_mode', 'deterministic_if_map')).strip().lower()
        if mode == 'deterministic' or (mode in ('deterministic_if_map', 'hybrid') and len(buildings) > 0):
            p_los = 1.0 if self.is_los(uav_pos, gt_pos, buildings=buildings) else 0.0
        else:
            p_los = self.calculate_los_probability(uav_pos, gt_pos, buildings)
        return float(p_los * pl_los + (1.0 - p_los) * pl_nlos)

    @staticmethod
    def _distance_3d_and_elevation_deg(tx_pos: np.ndarray, rx_pos: np.ndarray) -> Tuple[float, float]:
        """统一几何：三维距离 d 与仰角 θ（度）。"""
        tx_pos = np.asarray(tx_pos, dtype=float)
        rx_pos = np.asarray(rx_pos, dtype=float)
        d = float(np.linalg.norm(tx_pos - rx_pos))
        r = float(np.linalg.norm(tx_pos[:2] - rx_pos[:2]))
        h = float(abs(tx_pos[2] - rx_pos[2]))
        if r < 1e-9:
            theta = 90.0
        else:
            theta = float(np.degrees(np.arctan(h / r)))
        return max(d, 1e-6), theta

    def calculate_mean_path_loss_db(self,
                                    tx_pos: np.ndarray,
                                    rx_pos: np.ndarray,
                                    freq_hz: float,
                                    buildings: Optional[List[Building]] = None,
                                    force_nlos: bool = False,
                                    ignore_building_idx: Optional[int] = None,
                                    tx_rx_eta_los_db: Optional[float] = None,
                                    tx_rx_eta_nlos_db: Optional[float] = None) -> Tuple[float, float]:
        """
        统一链路预算：PL(dB)=P(LoS)*PL_LoS+(1-P(LoS))*PL_NLoS，并返回 (PL_dB, P_LoS)。
        - A2G/U2U 同一套公式，差别仅来自几何高度（tx_pos/rx_pos）。
        - 若 force_nlos=True 或被真实建筑遮挡，则 P_LoS=0。
        """
        tx_pos = np.asarray(tx_pos, dtype=float)
        rx_pos = np.asarray(rx_pos, dtype=float)
        d, _ = self._distance_3d_and_elevation_deg(tx_pos, rx_pos)

        buildings_list = buildings if buildings is not None else []
        blocked = False
        if buildings_list:
            if buildings_list is self.buildings:
                blocked = self._is_segment_blocked_by_buildings(tx_pos, rx_pos, ignore_building_idx=ignore_building_idx)
            else:
                blocked = self._is_segment_blocked_by_buildings_naive(tx_pos, rx_pos, buildings_list, ignore_building=None)

        if force_nlos or blocked:
            p_los = 0.0
        else:
            mode = str(getattr(self, 'los_indicator_mode', 'deterministic_if_map')).strip().lower()
            if mode == 'deterministic' or (mode in ('deterministic_if_map', 'hybrid') and len(buildings_list) > 0):
                p_los = 1.0
            else:
                p_los = float(self.calculate_los_probability(tx_pos, rx_pos, buildings_list))

        fspl = self.calculate_fspl_db_at_freq(d, float(freq_hz))
        eta_los = float(self.atg_model['eta_los_db'] if tx_rx_eta_los_db is None else tx_rx_eta_los_db)
        eta_nlos = float(self.atg_model['eta_nlos_db'] if tx_rx_eta_nlos_db is None else tx_rx_eta_nlos_db)
        pl_los = fspl + eta_los
        pl_nlos = fspl + eta_nlos
        pl = float(p_los * pl_los + (1.0 - p_los) * pl_nlos)
        return pl, float(p_los)

    def calculate_rx_power_dbm(self,
                               tx_power_dbm: float,
                               tx_pos: np.ndarray,
                               rx_pos: np.ndarray,
                               freq_hz: float,
                               buildings: Optional[List[Building]] = None,
                               force_nlos: bool = False,
                               ignore_building_idx: Optional[int] = None,
                               extra_loss_db: float = 0.0,
                               fading_gain_linear: float = 1.0,
                               duty_cycle_activity: float = 1.0) -> Tuple[float, float]:
        """
        统一接收功率模板（功率域第一公民）：
        P_rx[dBm] = P_tx[dBm] - PL_mean[dB] - extra_loss[dB] + 10log10(fading) + 10log10(activity)
        """
        pl_db, p_los = self.calculate_mean_path_loss_db(
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            freq_hz=float(freq_hz),
            buildings=buildings,
            force_nlos=force_nlos,
            ignore_building_idx=ignore_building_idx,
        )
        fade = max(float(fading_gain_linear), 1e-12)
        act = max(float(duty_cycle_activity), 0.0)
        act_db = 10.0 * np.log10(act) if act > 0.0 else -300.0
        rx_dbm = float(tx_power_dbm - pl_db - float(extra_loss_db) + 10.0 * np.log10(fade) + act_db)
        return rx_dbm, float(p_los)

    def _stable_gamma_sample(self, shape_k: float, scale_theta: float, key_values: Tuple[float, ...]) -> float:
        if shape_k <= 0.0 or scale_theta <= 0.0:
            return 0.0
        # deterministic sampling via seeded RNG
        u = self._stable_unit_float(*key_values)
        seed = int(max(0.0, min(1.0 - 1e-12, float(u))) * (2**31 - 1))
        rng = np.random.RandomState(seed)
        return float(rng.gamma(shape_k, scale_theta))

    def _nakagami_m_for_source(self, source: 'InterferenceSource', p_los: Optional[float] = None) -> float:
        type_id = int(getattr(source, 'type_id', -1))
        type_key = self.interference_config.type_mapping.get(type_id, 'unknown')
        m_by_type = getattr(self, 'nakagami_m_by_type', {}) if hasattr(self, 'nakagami_m_by_type') else {}
        m_default = float(getattr(self, 'nakagami_m_default', 1.0))
        m = float(m_by_type.get(type_key, m_default)) if isinstance(m_by_type, dict) else m_default
        m_los = getattr(self, 'nakagami_m_los', None)
        m_nlos = getattr(self, 'nakagami_m_nlos', None)
        if p_los is not None and (m_los is not None or m_nlos is not None):
            ml = float(m_los) if m_los is not None else m
            mn = float(m_nlos) if m_nlos is not None else m
            m = float(np.clip(float(p_los), 0.0, 1.0)) * ml + (1.0 - float(np.clip(float(p_los), 0.0, 1.0))) * mn
        m_min = float(getattr(self, 'nakagami_m_min', 0.5))
        m_max = float(getattr(self, 'nakagami_m_max', 20.0))
        return float(np.clip(m, m_min, m_max))

    def _nakagami_aggregate_interference_mw(self,
                                            drone_pos: np.ndarray,
                                            interference_sources: List['InterferenceSource'],
                                            buildings: Optional[List['Building']] = None,
                                            drone_idx: Optional[int] = None) -> float:
        if not interference_sources:
            return 0.0
        p1 = np.asarray(drone_pos, dtype=float)
        buildings_list = buildings if buildings is not None else []
        omegas = []
        ms = []
        for src_idx, source in enumerate(interference_sources):
            p0 = np.array([source.x, source.y, source.height], dtype=float)
            distance = float(np.linalg.norm(p0 - p1))
            path_loss_db, p_los = self._interference_path_loss_db(
                source=source,
                uav_pos=p1,
                buildings=buildings_list,
                p0=p0,
                distance_m=distance,
                blocked=None,
            )
            effective_power_dbm = float(source.power - float(path_loss_db))
            if distance > float(source.coverage_radius) and float(source.coverage_radius) > 0.0:
                excess_ratio = float(distance / float(source.coverage_radius))
                effective_power_dbm -= float(20.0 * np.log10(max(excess_ratio, 1e-6)))
            freq_coupling = self._spectral_coupling_factor(
                tx_freq_hz=float(source.frequency),
                rx_freq_hz=float(self.communication_freq),
                rx_bw_hz=float(getattr(self, 'bandwidth_hz', 20e6)),
                interferer_type_id=int(getattr(source, 'type_id', -1)),
            )
            activity = float(getattr(source, 'activity', 1.0))
            omega = float(self._dbm_to_mw(effective_power_dbm)) * float(freq_coupling) * float(activity)
            if omega <= 0.0:
                continue
            m_i = self._nakagami_m_for_source(source, p_los=p_los)
            omegas.append(float(omega))
            ms.append(float(m_i))
        if not omegas:
            return 0.0
        omega_sum = float(np.sum(omegas))
        if omega_sum <= 0.0:
            return 0.0
        var_sum = 0.0
        for omega_i, m_i in zip(omegas, ms):
            m_i = max(float(m_i), 1e-6)
            var_sum += float(omega_i) ** 2 / m_i
        if var_sum <= 0.0:
            return omega_sum
        m_eq = float(omega_sum ** 2 / var_sum)
        m_eq = float(np.clip(m_eq, getattr(self, 'nakagami_m_min', 0.5), getattr(self, 'nakagami_m_max', 20.0)))
        if str(getattr(self, 'nakagami_sample_mode', 'sample')).strip().lower() in ('mean', 'avg', 'expectation'):
            return omega_sum
        scale = float(omega_sum / max(m_eq, 1e-12))
        key = (
            23.0,
            float(drone_idx if drone_idx is not None else 0.0),
            float(p1[0]), float(p1[1]), float(p1[2]),
            float(omega_sum), float(m_eq),
        )
        return float(self._stable_gamma_sample(m_eq, scale, key))

    def _bs_power_total_w_static(self, scenario: Dict) -> float:
        if not bool(getattr(self, 'bs_energy_enabled', False)):
            return 0.0
        n = int(getattr(self, 'num_stations', 1))
        if n <= 0:
            return 0.0
        alpha = float(self._traffic_profile_alpha_avg())
        users_list = self._resolve_users_per_station(scenario, 0, 0.0)
        caps = self._resolve_capacity_per_station(scenario)
        bs_types = scenario.get('bs_types', None)
        total_w = 0.0
        for i in range(n):
            bs_type = self.bs_type
            if isinstance(bs_types, (list, tuple)) and len(bs_types) == n:
                bs_type = str(bs_types[i])
            params = self._get_bs_params(bs_type)
            cap_i = float(caps[i]) if i < len(caps) else float(caps[-1])
            u_i = float(users_list[i]) if i < len(users_list) else float(users_list[-1])
            cap_i = max(cap_i, 1e-9)
            load = max(0.0, (u_i / cap_i) * alpha)
            load = min(1.0, load)
            if load > 0.0:
                load = max(load, float(getattr(self, 'bs_load_floor', 0.0)))
            p_out = float(load) * float(params.get('p_max_w', 0.0))
            total_w += float(self._bs_power_input_w(p_out, params))
        return float(total_w)


    def calculate_interference_power_mw(self,
                                        drone_pos: np.ndarray,
                                        interference_sources: List[InterferenceSource],
                                        buildings: Optional[List[Building]] = None,
                                        drone_idx: Optional[int] = None) -> float:
        if not interference_sources:
            return 0.0

        model = str(getattr(self, 'interference_aggregation_model', 'linear')).strip().lower()
        if model in ('nakagami', 'nakagami_m', 'moment', 'mm'):
            return float(self._nakagami_aggregate_interference_mw(
                drone_pos=drone_pos,
                interference_sources=interference_sources,
                buildings=buildings,
                drone_idx=drone_idx,
            ))

        total_mw = 0.0
        p1 = np.asarray(drone_pos, dtype=float)
        buildings_list = buildings if buildings is not None else []
        for src_idx, source in enumerate(interference_sources):
            p0 = np.array([source.x, source.y, source.height], dtype=float)
            distance = float(np.linalg.norm(p0 - p1))
            path_loss_db, _ = self._interference_path_loss_db(
                source=source,
                uav_pos=p1,
                buildings=buildings_list,
                p0=p0,
                distance_m=distance,
                blocked=None,
            )
            effective_power_dbm = float(source.power - float(path_loss_db))

            if distance > float(source.coverage_radius) and float(source.coverage_radius) > 0.0:
                excess_ratio = float(distance / float(source.coverage_radius))
                effective_power_dbm -= float(20.0 * np.log10(max(excess_ratio, 1e-6)))

            freq_coupling = self._spectral_coupling_factor(
                tx_freq_hz=float(source.frequency),
                rx_freq_hz=float(self.communication_freq),
                rx_bw_hz=float(getattr(self, 'bandwidth_hz', 20e6)),
                interferer_type_id=int(getattr(source, 'type_id', -1)),
            )
            fading_gain = self._stable_exp_fading_gain(int(drone_idx) if drone_idx is not None else 0, int(src_idx), source)
            activity = float(getattr(source, 'activity', 1.0))
            total_mw += self._dbm_to_mw(effective_power_dbm) * float(freq_coupling) * float(fading_gain) * activity

        return float(total_mw)

    def calculate_power_intensity_change(self,
                                         drone_pos: np.ndarray,
                                         station_pos: np.ndarray,
                                         scenario: Dict,
                                         drone_idx: Optional[int] = None) -> float:
        distance = np.linalg.norm(drone_pos - station_pos)
        if distance < 1.0:
            distance = 1.0

        path_loss_db = self.calculate_atg_mean_path_loss_db(drone_pos, station_pos, scenario['buildings'])
        rx_signal_dbm = self._comm_tx_power_dbm - path_loss_db
        rx_signal_mw = self._dbm_to_mw(rx_signal_dbm)

        baseline_rx_mw = float(self._baseline_rx_mw)

        interference_mw: Optional[float] = None
        if drone_idx is not None and 'interference_power_mw_per_drone' in scenario:
            try:
                interference_mw = float(scenario['interference_power_mw_per_drone'][int(drone_idx)])
            except Exception:
                interference_mw = None

        if interference_mw is None:
            interference_mw = self.calculate_interference_power_mw(
                drone_pos,
                scenario.get('interference_sources', []),
                buildings=scenario.get('buildings', []),
                drone_idx=drone_idx
            )

        eps = 1e-12
        power_increase_ratio = (baseline_rx_mw / (rx_signal_mw + eps)) * (1.0 + interference_mw / (rx_signal_mw + eps))
        power_increase_ratio = max(power_increase_ratio, 1.0)

        ratio_db = 10.0 * np.log10(power_increase_ratio)
        scale_db = float(self._power_change_scale_db)
        if ratio_db <= 0.0:
            return 0.0

        return float(np.clip(1.0 - np.exp(-ratio_db / max(scale_db, 1e-6)), 0.0, 1.0))

    def _noise_power_mw(self) -> float:
        """噪声功率 Ny（mW）：N0[dBm/Hz] + 10log10(B) + NF。"""
        b = max(float(getattr(self, 'bandwidth_hz', 1.0)), 1.0)
        n0_dbm_hz = float(getattr(self, 'thermal_noise_dbm_hz', -174.0))
        nf_db = float(getattr(self, 'noise_figure_db', 0.0))
        noise_dbm = float(n0_dbm_hz + 10.0 * np.log10(b) + nf_db)
        return float(self._dbm_to_mw(noise_dbm))

    def calculate_power_margin_components(self,
                                         drone_pos: np.ndarray,
                                         station_pos: np.ndarray,
                                         scenario: Dict,
                                         drone_idx: Optional[int] = None) -> Dict[str, float]:
        """
        统一功率域分解（不依赖“比值口径”的 SINR）：
        - P_rx[dBm]：接收信号功率
        - I_y[mW]：总干扰功率（线性累加）
        - N_y[mW]：噪声功率（线性）
        - P_IN[dBm]：干扰+噪声的等效功率
        - M[dB] = P_rx[dBm] - P_IN[dBm]：功率裕量（等价于 SINR 的 dB 形式）
        - SINR = 10^(M/10)（可选输出）
        - R = B log2(1 + 10^(M/10))（可选输出）
        """
        drone_pos = np.asarray(drone_pos, dtype=float)
        station_pos = np.asarray(station_pos, dtype=float)

        # 约定：A2G 链路为 “station -> drone”（地面站发射、无人机接收），干扰聚合也在无人机端计算
        buildings_list = scenario.get('buildings', [])
        fade = self._stable_exp_fading_gain_for_link((
            9.0,
            float(drone_idx if drone_idx is not None else 0),
            float(drone_pos[0]), float(drone_pos[1]), float(drone_pos[2]),
            float(station_pos[0]), float(station_pos[1]), float(station_pos[2]),
        ))
        prx_dbm, p_los = self.calculate_rx_power_dbm(
            tx_power_dbm=float(self._comm_tx_power_dbm),
            tx_pos=station_pos,
            rx_pos=drone_pos,
            freq_hz=float(self.communication_freq),
            buildings=buildings_list,
            force_nlos=False,
            ignore_building_idx=None,
            extra_loss_db=0.0,
            fading_gain_linear=float(fade),
            duty_cycle_activity=1.0,
        )
        prx_mw = float(self._dbm_to_mw(prx_dbm))
        # PL 便于阈值事件：P(PL > PL_max)
        pl_db, _ = self.calculate_mean_path_loss_db(
            tx_pos=station_pos, rx_pos=drone_pos, freq_hz=float(self.communication_freq), buildings=buildings_list
        )

        # 干扰：I_y（mW）
        iy_mw: Optional[float] = None
        if drone_idx is not None and 'interference_power_mw_per_drone' in scenario:
            try:
                iy_mw = float(scenario['interference_power_mw_per_drone'][int(drone_idx)])
            except Exception:
                iy_mw = None
        if iy_mw is None:
            iy_mw = float(self.calculate_interference_power_mw(
                drone_pos=drone_pos,
                interference_sources=scenario.get('interference_sources', []),
                buildings=buildings_list,
                drone_idx=drone_idx,
            ))

        # 噪声：N_y（mW）
        ny_mw = float(self._noise_power_mw())
        pin_mw = float(max(iy_mw, 0.0) + max(ny_mw, 0.0))
        pin_dbm = float(self._mw_to_dbm(pin_mw))
        n_dbm = float(self._mw_to_dbm(ny_mw))
        delta_in_db = float(pin_dbm - n_dbm)
        iy_dbm = float(self._mw_to_dbm(max(float(iy_mw), 0.0)))

        m_db = float(prx_dbm - pin_dbm)
        sinr_lin = float(10.0 ** (m_db / 10.0))

        b = max(float(getattr(self, 'bandwidth_hz', 1.0)), 1.0)
        rate_bps = float(b * np.log2(1.0 + sinr_lin))

        outage = 1.0 if m_db < float(getattr(self, 'margin_threshold_db', 0.0)) else 0.0
        return {
            'P_rx_dbm': prx_dbm,
            'P_rx_mw': prx_mw,
            'I_y_mw': float(iy_mw),
            'I_y_dbm': iy_dbm,
            'N_y_mw': ny_mw,
            'N_y_dbm': n_dbm,
            'P_IN_dbm': pin_dbm,
            'Delta_IN_db': delta_in_db,
            'M_db': m_db,
            'SINR_linear': sinr_lin,
            'R_bps': rate_bps,
            'Outage': outage,
            'PL_db': float(pl_db),
            'P_LoS': float(p_los),
        }

    def _cdi_intensity_mapping_db(self, interference_mw: float) -> float:
        """g(I)：把“聚合干扰强度 I（mW）”映射到 dB-like 尺度，避免直接用比值/SINR。"""
        i_mw = max(float(interference_mw), 0.0)
        i_ref = max(float(getattr(self, 'cdi_int_ref_mw', 1e-12)), 1e-12)
        return float(10.0 * np.log10(1.0 + i_mw / i_ref))

    def calculate_cdi_components(self,
                                 drone_pos: np.ndarray,
                                 station_pos: np.ndarray,
                                 scenario: Dict,
                                 drone_idx: Optional[int] = None) -> Tuple[float, float, float]:
        """
        统一通信劣化指数（CDI）分量：
        - D_link = max(0, Λ - PL_max)  （只用路径损耗）
        - D_int  = I = Σ h_i l(r_i)   （聚合干扰强度，线性累加）
        - CDI    = w_L·D_link + w_I·g(D_int)
        """
        lambda_db = float(self.calculate_atg_mean_path_loss_db(drone_pos, station_pos, scenario.get('buildings', [])))
        d_link = float(max(0.0, lambda_db - float(self.cdi_pl_max_db)))

        interference_mw: Optional[float] = None
        if drone_idx is not None and 'interference_power_mw_per_drone' in scenario:
            try:
                interference_mw = float(scenario['interference_power_mw_per_drone'][int(drone_idx)])
            except Exception:
                interference_mw = None
        if interference_mw is None:
            interference_mw = float(self.calculate_interference_power_mw(
                drone_pos=np.asarray(drone_pos, dtype=float),
                interference_sources=scenario.get('interference_sources', []),
                buildings=scenario.get('buildings', []),
                drone_idx=drone_idx,
            ))

        g_int_db = self._cdi_intensity_mapping_db(interference_mw)
        cdi = float(self.cdi_w_link * d_link + self.cdi_w_int * g_int_db)
        return d_link, float(interference_mw), cdi

    def calculate_comm_degradation(self,
                                   drone_pos: np.ndarray,
                                   station_pos: np.ndarray,
                                   scenario: Dict,
                                   drone_idx: Optional[int] = None) -> float:
        """
        通信劣化程度 ∈[0,1]：统一使用功率裕量（margin）模型计算。
        """
        model = str(getattr(self, 'degradation_model', 'margin')).strip().lower()
        if model in ('legacy', 'heuristic', 'old', 'power_ratio'):
            base = float(self.calculate_power_intensity_change(drone_pos, station_pos, scenario, drone_idx=drone_idx))
            return float(self._blend_u2u_comm_degradation(base, scenario, drone_idx=drone_idx))

        if model in ('margin', 'power_margin', 'm'):
            comp = self.calculate_power_margin_components(drone_pos, station_pos, scenario, drone_idx=drone_idx)
            m_db = float(comp['M_db'])
            theta = float(getattr(self, 'margin_threshold_db', 0.0))
            slope = max(float(getattr(self, 'margin_slope_db', 1.0)), 1e-6)
            # comm_deg 越大表示越差：当 M 低于门限时快速趋近 1
            comm_deg = 1.0 / (1.0 + np.exp((m_db - theta) / slope))
            return float(self._blend_u2u_comm_degradation(float(np.clip(comm_deg, 0.0, 1.0)), scenario, drone_idx=drone_idx))

        # CDI 分支停用：兜底也统一回落到 margin（功率裕量）模型
        comp = self.calculate_power_margin_components(drone_pos, station_pos, scenario, drone_idx=drone_idx)
        m_db = float(comp['M_db'])
        theta = float(getattr(self, 'margin_threshold_db', 0.0))
        slope = max(float(getattr(self, 'margin_slope_db', 1.0)), 1e-6)
        comm_deg = 1.0 / (1.0 + np.exp((m_db - theta) / slope))
        return float(self._blend_u2u_comm_degradation(float(np.clip(comm_deg, 0.0, 1.0)), scenario, drone_idx=drone_idx))

    def _comm_deg_from_margin_db(self, m_db: float) -> float:
        """将功率裕量 M[dB] 映射为通信劣化程度∈[0,1]（越大越差）。"""
        theta = float(getattr(self, 'margin_threshold_db', 0.0))
        slope = max(float(getattr(self, 'margin_slope_db', 1.0)), 1e-6)
        comm_deg = 1.0 / (1.0 + np.exp((float(m_db) - theta) / slope))
        return float(np.clip(comm_deg, 0.0, 1.0))

    def _u2u_comm_degradation_for_drone(self, scenario: Dict, drone_idx: int) -> Optional[float]:
        """
        UAV↔UAV 通信链路的“有用信道”劣化（不作为干扰 I_aer）。

        计算方式：复用 U2U 的功率预算（calculate_u2u_power_margin_components），
        取与该 UAV 相关的所有 U2U 方向链路的劣化并聚合（mean/max 可选）。
        """
        if not getattr(self, 'u2u_enabled', False):
            return None

        drones = np.asarray(scenario.get('drone_positions', np.zeros((0, 3))), dtype=float)
        if int(drone_idx) < 0 or int(drone_idx) >= len(drones) or len(drones) < 2:
            return None

        agg = str(self.user_params.get('u2u_comm_aggregate', 'mean')).strip().lower()
        vals: List[float] = []

        it = self._iter_u2u_links(scenario)
        if it is None:
            return None

        for i, j, pi, pj in it:
            # 对于每个无向 pair，计算两个方向（i->j 与 j->i），并把涉及 drone_idx 的方向计入
            if int(drone_idx) == int(i):
                pm_ij = self.calculate_u2u_power_margin_components(pi, pj, scenario, tx_idx=int(i), rx_idx=int(j))
                vals.append(self._comm_deg_from_margin_db(float(pm_ij.get('M_db', 0.0))))
                pm_ji = self.calculate_u2u_power_margin_components(pj, pi, scenario, tx_idx=int(j), rx_idx=int(i))
                vals.append(self._comm_deg_from_margin_db(float(pm_ji.get('M_db', 0.0))))
            elif int(drone_idx) == int(j):
                pm_ij = self.calculate_u2u_power_margin_components(pi, pj, scenario, tx_idx=int(i), rx_idx=int(j))
                vals.append(self._comm_deg_from_margin_db(float(pm_ij.get('M_db', 0.0))))
                pm_ji = self.calculate_u2u_power_margin_components(pj, pi, scenario, tx_idx=int(j), rx_idx=int(i))
                vals.append(self._comm_deg_from_margin_db(float(pm_ji.get('M_db', 0.0))))

        if not vals:
            return None

        if agg in ('max', 'worst', 'worst_case'):
            return float(max(vals))
        return float(np.mean(vals))

    def _blend_u2u_comm_degradation(self, a2g_comm_deg: float, scenario: Dict, drone_idx: Optional[int]) -> float:
        """
        将 U2U 链路损耗“纳入通信劣化评估”。

        说明：UAV↔UAV 是“有用信道”，因此不进入干扰项；其链路质量通过此处融合到 comm_deg。
        """
        base = float(np.clip(a2g_comm_deg, 0.0, 1.0))
        if drone_idx is None or not getattr(self, 'u2u_enabled', False):
            return base

        u2u_deg = self._u2u_comm_degradation_for_drone(scenario, int(drone_idx))
        if u2u_deg is None:
            return base

        w = self.user_params.get('u2u_comm_weight', None)
        if w is None:
            # 默认：启用 U2U 时，A2G 与 U2U 各占一半
            w_u2u = 0.5
        else:
            w_u2u = float(w)
        w_u2u = float(np.clip(w_u2u, 0.0, 1.0))

        return float(np.clip((1.0 - w_u2u) * base + w_u2u * float(u2u_deg), 0.0, 1.0))

    def calculate_speed_energy_efficiency_degradation(self, drone_speed_mps: float) -> float:
        """
        速度导致的通信能效劣化（基于旋翼无人机推进功率模型 P(V)）。

        使用图中公式（blade profile + induced + parasite），将“能效”近似为单位时间可用能量的倒数：
        - 以悬停功率 P(0)=P0+Pi 为参考，速度能效劣化 = max(0, 1 - P(0)/P(V))，范围[0,1]。
        """
        v = max(float(drone_speed_mps), 0.0)

        P0 = float(self.power_model['rotor_P0_w'])
        Pi = float(self.power_model['rotor_Pi_w'])
        U_tip = max(float(self.power_model['rotor_U_tip_mps']), 1e-6)
        v0 = max(float(self.power_model['rotor_v0_mps']), 1e-6)
        d0 = float(self.power_model['rotor_d0'])
        rho = float(self.power_model['air_rho'])
        s = float(self.power_model['rotor_s'])
        A = float(self.power_model['rotor_A_m2'])

        # P(V)=P0(1+3V^2/U_tip^2) + Pi*sqrt( sqrt(1+V^4/(4v0^4)) - V^2/(2v0^2) ) + 1/2*d0*rho*s*A*V^3
        blade_profile = P0 * (1.0 + 3.0 * (v ** 2) / (U_tip ** 2))
        induced_inner = np.sqrt(1.0 + (v ** 4) / (4.0 * (v0 ** 4))) - (v ** 2) / (2.0 * (v0 ** 2))
        induced = Pi * np.sqrt(max(float(induced_inner), 0.0))
        parasite = 0.5 * d0 * rho * s * A * (v ** 3)

        propulsion_power_w = float(blade_profile + induced + parasite)
        hover_power_w = float(P0 + Pi)

        propulsion_power_w = max(propulsion_power_w, 1e-6)
        hover_power_w = max(hover_power_w, 1e-6)

        degradation = 1.0 - (hover_power_w / propulsion_power_w)
        return float(np.clip(max(0.0, degradation), 0.0, 1.0))

    def _rotor_propulsion_power_w(self, drone_speed_mps: float) -> float:
        """旋翼无人机推进功率 P(V)（W），用于 bit/J 能效计算。"""
        v = max(float(drone_speed_mps), 0.0)
        P0 = float(self.power_model['rotor_P0_w'])
        Pi = float(self.power_model['rotor_Pi_w'])
        U_tip = max(float(self.power_model['rotor_U_tip_mps']), 1e-6)
        v0 = max(float(self.power_model['rotor_v0_mps']), 1e-6)
        d0 = float(self.power_model['rotor_d0'])
        rho = float(self.power_model['air_rho'])
        s = float(self.power_model['rotor_s'])
        A = float(self.power_model['rotor_A_m2'])

        blade_profile = P0 * (1.0 + 3.0 * (v ** 2) / (U_tip ** 2))
        induced_inner = np.sqrt(1.0 + (v ** 4) / (4.0 * (v0 ** 4))) - (v ** 2) / (2.0 * (v0 ** 2))
        induced = Pi * np.sqrt(max(float(induced_inner), 0.0))
        parasite = 0.5 * d0 * rho * s * A * (v ** 3)
        return float(max(blade_profile + induced + parasite, 1e-6))

    def calculate_energy_efficiency_metrics(self, scenario: Dict) -> Dict[str, float]:
        """
        能效 EE（bit/J）：EE = (Σ_k R_k) / (Σ_k P_prop(V_k))
        这里用“无人机 k 对所有地面站的平均速率 R_k”来近似其吞吐。
        """
        if bool(getattr(self, 'tdma_enabled', False)):
            return self.calculate_tdma_energy_efficiency_metrics(scenario)
        drones = np.asarray(scenario.get('drone_positions', np.zeros((0, 3))), dtype=float)
        stations = np.asarray(scenario.get('station_positions', np.zeros((0, 3))), dtype=float)
        speeds = np.asarray(scenario.get('drone_speeds', np.zeros((len(drones),))), dtype=float)
        if drones.size == 0 or stations.size == 0:
            return {'sum_rate_bps': 0.0, 'sum_propulsion_w': 0.0, 'EE_bpj': 0.0}

        sum_rate_bps = 0.0
        sum_prop_w = 0.0
        for drone_idx in range(int(len(drones))):
            rate_list = []
            for station_idx in range(int(len(stations))):
                comp = self.calculate_power_margin_components(
                    drones[drone_idx],
                    stations[station_idx],
                    scenario,
                    drone_idx=int(drone_idx),
                )
                rate_list.append(float(comp.get('R_bps', 0.0)))
            mean_rate_bps = float(np.mean(rate_list)) if rate_list else 0.0
            v = float(speeds[drone_idx]) if drone_idx < len(speeds) else 0.0
            sum_rate_bps += mean_rate_bps
            sum_prop_w += self._rotor_propulsion_power_w(v)

        bs_power_w = 0.0
        if bool(getattr(self, 'include_bs_energy_in_ee', False)):
            bs_power_w = float(self._bs_power_total_w_static(scenario))
        sum_prop_w += float(bs_power_w)
        ee_bpj = float(sum_rate_bps / max(sum_prop_w, 1e-12))
        return {'sum_rate_bps': float(sum_rate_bps), 'sum_propulsion_w': float(sum_prop_w), 'EE_bpj': ee_bpj}


    def _get_bs_params(self, bs_type: str) -> Dict[str, float]:
        params = dict(self.EARTH_BS_PARAMS.get(str(bs_type).lower(), self.EARTH_BS_PARAMS.get('micro')))
        override = dict(getattr(self, 'bs_params_override', {}))
        if str(bs_type).lower() in override and isinstance(override.get(str(bs_type).lower()), dict):
            params.update(override.get(str(bs_type).lower(), {}))
        else:
            for k in ('p_max_w', 'n_trx', 'p0_w', 'delta_p', 'p_sleep_w'):
                if k in override:
                    params[k] = override[k]
        return params

    def _traffic_profile_alpha(self, t_s: float) -> float:
        # t_s in seconds, mapped to hour-of-day
        hour = (float(t_s) + float(getattr(self, 'bs_time_origin_s', 0.0))) / 3600.0
        hour = hour % 24.0
        hourly = getattr(self, 'bs_traffic_profile_hourly', None)
        points = getattr(self, 'bs_traffic_profile_points', None)
        if isinstance(hourly, (list, tuple)) and len(hourly) >= 2:
            h0 = int(math.floor(hour)) % len(hourly)
            h1 = (h0 + 1) % len(hourly)
            frac = hour - math.floor(hour)
            a = (1.0 - frac) * float(hourly[h0]) + float(frac) * float(hourly[h1])
            return float(np.clip(a, 0.0, 1.0))
        if isinstance(points, (list, tuple)) and len(points) >= 2:
            pts = []
            for p in points:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((float(p[0]) % 24.0, float(p[1])))
            pts = sorted(pts, key=lambda x: x[0])
            if len(pts) >= 2:
                # wrap around 24h
                pts_wrap = pts + [(pts[0][0] + 24.0, pts[0][1])]
                for i in range(len(pts_wrap) - 1):
                    h0, a0 = pts_wrap[i]
                    h1, a1 = pts_wrap[i + 1]
                    h = hour if hour >= h0 else hour + 24.0
                    if h0 <= h <= h1:
                        frac = 0.0 if h1 == h0 else (h - h0) / (h1 - h0)
                        a = (1.0 - frac) * a0 + frac * a1
                        return float(np.clip(a, 0.0, 1.0))
        # default daily profile (normalized)
        default_hourly = [
            0.15, 0.12, 0.10, 0.08, 0.08, 0.12,
            0.25, 0.45, 0.60, 0.70, 0.75, 0.80,
            0.85, 0.80, 0.75, 0.70, 0.80, 0.95,
            1.00, 0.90, 0.70, 0.50, 0.35, 0.25,
        ]
        h0 = int(math.floor(hour)) % 24
        h1 = (h0 + 1) % 24
        frac = hour - math.floor(hour)
        a = (1.0 - frac) * float(default_hourly[h0]) + float(frac) * float(default_hourly[h1])
        return float(np.clip(a, 0.0, 1.0))

    def _traffic_profile_alpha_avg(self) -> float:
        hourly = getattr(self, 'bs_traffic_profile_hourly', None)
        if isinstance(hourly, (list, tuple)) and len(hourly) > 0:
            vals = [float(v) for v in hourly]
            return float(np.clip(float(np.mean(vals)), 0.0, 1.0))
        points = getattr(self, 'bs_traffic_profile_points', None)
        if isinstance(points, (list, tuple)) and len(points) >= 2:
            # sample 24 points
            vals = []
            for h in range(24):
                vals.append(self._traffic_profile_alpha(float(h) * 3600.0))
            return float(np.clip(float(np.mean(vals)), 0.0, 1.0))
        return float(np.clip(float(np.mean([
            0.15, 0.12, 0.10, 0.08, 0.08, 0.12,
            0.25, 0.45, 0.60, 0.70, 0.75, 0.80,
            0.85, 0.80, 0.75, 0.70, 0.80, 0.95,
            1.00, 0.90, 0.70, 0.50, 0.35, 0.25,
        ])), 0.0, 1.0))

    def _resolve_users_per_station(self, scenario: Dict, t_idx: int, t_s: float) -> List[float]:
        n = int(getattr(self, 'num_stations', 1))
        if n <= 0:
            return [0.0]
        users_ts = scenario.get('users_per_station_time_series', None)
        if isinstance(users_ts, (list, tuple)) and len(users_ts) > 0:
            u = users_ts[min(int(t_idx), len(users_ts) - 1)]
            if isinstance(u, (list, tuple)) and len(u) == n:
                return [float(x) for x in u]
            if isinstance(u, (int, float)):
                return [float(u)] * n
        users_total_ts = scenario.get('users_time_series', None)
        if isinstance(users_total_ts, (list, tuple)) and len(users_total_ts) > 0:
            u_total = float(users_total_ts[min(int(t_idx), len(users_total_ts) - 1)])
            return [float(u_total) / max(float(n), 1.0)] * n
        users_per_station = scenario.get('users_per_station', None)
        if isinstance(users_per_station, (list, tuple)) and len(users_per_station) == n:
            return [float(x) for x in users_per_station]
        if isinstance(users_per_station, (int, float)):
            return [float(users_per_station)] * n
        users_total = scenario.get('users_total', None)
        if isinstance(users_total, (int, float)):
            return [float(users_total) / max(float(n), 1.0)] * n
        return [float(getattr(self, 'bs_users_per_station', 0.0))] * n

    def _resolve_capacity_per_station(self, scenario: Dict) -> List[float]:
        n = int(getattr(self, 'num_stations', 1))
        cap_ts = scenario.get('capacity_per_station', None)
        if isinstance(cap_ts, (list, tuple)) and len(cap_ts) == n:
            return [float(x) for x in cap_ts]
        cap = scenario.get('bs_capacity_max', scenario.get('capacity_max', getattr(self, 'bs_capacity_max', 100.0)))
        if isinstance(cap, (int, float)):
            return [float(cap)] * max(n, 1)
        return [float(getattr(self, 'bs_capacity_max', 100.0))] * max(n, 1)

    def _bs_power_input_w(self, p_out_w: float, params: Dict[str, float]) -> float:
        n_trx = float(params.get('n_trx', 1))
        p0_w = float(params.get('p0_w', 0.0))
        delta_p = float(params.get('delta_p', 0.0))
        p_sleep_w = float(params.get('p_sleep_w', 0.0))
        if float(p_out_w) <= 0.0 and bool(getattr(self, 'bs_sleep_enabled', True)):
            return float(n_trx * p_sleep_w)
        return float(n_trx * (p0_w + delta_p * float(p_out_w)))

    def _bs_power_total_w_at_time(self, scenario: Dict, t_idx: int, t_s: float) -> float:
        if not bool(getattr(self, 'bs_energy_enabled', False)):
            return 0.0
        n = int(getattr(self, 'num_stations', 1))
        if n <= 0:
            return 0.0
        alpha = float(self._traffic_profile_alpha(float(t_s)))
        users_list = self._resolve_users_per_station(scenario, int(t_idx), float(t_s))
        caps = self._resolve_capacity_per_station(scenario)
        bs_types = scenario.get('bs_types', None)
        total_w = 0.0
        for i in range(n):
            bs_type = self.bs_type
            if isinstance(bs_types, (list, tuple)) and len(bs_types) == n:
                bs_type = str(bs_types[i])
            params = self._get_bs_params(bs_type)
            cap_i = float(caps[i]) if i < len(caps) else float(caps[-1])
            u_i = float(users_list[i]) if i < len(users_list) else float(users_list[-1])
            cap_i = max(cap_i, 1e-9)
            load = max(0.0, (u_i / cap_i) * alpha)
            load = min(1.0, load)
            if load > 0.0:
                load = max(load, float(getattr(self, 'bs_load_floor', 0.0)))
            p_out = float(load) * float(params.get('p_max_w', 0.0))
            total_w += float(self._bs_power_input_w(p_out, params))
        return float(total_w)

    def _build_time_grid(self) -> Tuple[np.ndarray, float]:
        """
        离散时间网格：t[n]=n·Δt，n=0..N-1。
        用于把连续时间积分近似为求和（TDMA 吞吐/能量）。
        """
        t_total = float(self.user_params.get('traj_total_time_s', 20.0))
        dt = float(self.user_params.get('traj_dt_s', 1.0))
        t_total = max(t_total, 1e-6)
        dt = max(dt, 1e-6)
        n = int(max(1, round(t_total / dt)))
        t = np.arange(n, dtype=float) * dt
        return t, float(dt)

    def _trajectory_positions_discrete(self,
                                       drone_idx: int,
                                       scenario: Dict,
                                       t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        给出离散时间轨迹 q[n] 与速度标量 V[n]。
        - hover: q 固定，V=0
        - orbit: q 围绕初始点做圆周，速度由染色体 speed 给出（V=rw）
        - static: 不改变位置（保持现有“静态场景”口径），V 用染色体 speed（仅影响能耗）
        """
        drones0 = np.asarray(scenario.get('drone_positions', np.zeros((0, 3))), dtype=float)
        speeds0 = np.asarray(scenario.get('drone_speeds', np.zeros((len(drones0),))), dtype=float)
        if drones0.size == 0 or int(drone_idx) < 0 or int(drone_idx) >= len(drones0):
            return np.zeros((len(t), 3), dtype=float), np.zeros((len(t),), dtype=float)

        p0 = np.asarray(drones0[int(drone_idx)], dtype=float)
        v0 = float(speeds0[int(drone_idx)]) if int(drone_idx) < len(speeds0) else 0.0
        mode = str(self.user_params.get('traj_mode', 'static')).strip().lower()
        # 便捷映射：速度为 0 -> hover
        if v0 <= 1e-6 and mode in ('orbit', 'orbit_from_speed', 'auto', 'mode_by_speed'):
            mode = 'hover'

        pos = np.zeros((len(t), 3), dtype=float)
        spd = np.zeros((len(t),), dtype=float)

        if mode in ('hover',):
            pos[:] = p0
            spd[:] = 0.0
            return pos, spd

        if mode in ('orbit', 'orbit_from_speed'):
            r = float(self.user_params.get('orbit_radius_m', 30.0))
            r = max(r, 1e-3)
            w = float(v0 / r)  # rad/s
            theta0 = float(self._stable_unit_float(31.0, float(p0[0]), float(p0[1]), float(p0[2])) * 2.0 * np.pi)
            pos[:, 0] = float(p0[0]) + r * np.cos(w * t + theta0)
            pos[:, 1] = float(p0[1]) + r * np.sin(w * t + theta0)
            pos[:, 2] = float(p0[2])
            spd[:] = float(abs(v0))
            return pos, spd

        # static（默认）：q 不随时间变化，speed 仅用于 P(V)
        pos[:] = p0
        spd[:] = float(abs(v0))
        return pos, spd

    def calculate_tdma_energy_efficiency_metrics(self, scenario: Dict) -> Dict[str, float]:
        """
        离散 TDMA：对每个 UAV，在每个时间步只服务一个 GN（这里用 ground stations 作为 GN）。

        - 速率：R_k[n] = B log2(1 + SINR/Γ)
        - TDMA：λ_k[n]∈{0,1}，每步选择一个 GN：argmax_k R_k[n]
        - 吞吐：R~_k = Σ_n λ_k[n] R_k[n] Δt
        - 能量：E = Σ_n P(V[n])Δt + Pc Σ_n (Σ_k λ_k[n])Δt
        - 能效：EE = (Σ_k R~_k) / E
        """
        drones = np.asarray(scenario.get('drone_positions', np.zeros((0, 3))), dtype=float)
        stations = np.asarray(scenario.get('station_positions', np.zeros((0, 3))), dtype=float)
        if drones.size == 0 or stations.size == 0:
            return {'sum_rate_bps': 0.0, 'sum_propulsion_w': 0.0, 'EE_bpj': 0.0, 'tdma_enabled': 1.0}

        t, dt = self._build_time_grid()
        b = max(float(getattr(self, 'bandwidth_hz', 1.0)), 1.0)
        gamma = float(self.user_params.get('coding_gap_linear', 1.0))
        gamma = max(gamma, 1.0)
        pc = float(self.user_params.get('comm_circuit_power_w', 0.0))
        pc = max(pc, 0.0)

        total_bits = 0.0
        total_energy_j = 0.0
        # BS energy over time (EARTH model)
        if bool(getattr(self, 'include_bs_energy_in_ee', False)) and bool(getattr(self, 'bs_energy_enabled', False)):
            bs_energy_j = 0.0
            for n_idx, t_s in enumerate(t):
                bs_energy_j += float(self._bs_power_total_w_at_time(scenario, int(n_idx), float(t_s))) * float(dt)
            total_energy_j += float(bs_energy_j)
        station_bits = np.zeros((len(stations),), dtype=float)

        # 轨迹评估不复用静态“每无人机干扰缓存”，避免位置变化导致口径错误
        scenario_no_cache = dict(scenario)
        scenario_no_cache.pop('interference_power_mw_per_drone', None)

        for di in range(int(len(drones))):
            traj, spd = self._trajectory_positions_discrete(int(di), scenario, t)
            # 推进能量
            e_prop = 0.0
            for vn in spd:
                e_prop += float(self._rotor_propulsion_power_w(float(vn))) * float(dt)
            total_energy_j += float(e_prop)

            # TDMA 通信：每步选一个 GN
            comm_time = 0.0
            for n in range(len(t)):
                rates = []
                for si in range(int(len(stations))):
                    drone_pos_n = traj[n]
                    station_pos = stations[si]
                    # 不使用预计算缓存（轨迹会变），每步按当前位置重算 I/N
                    pm = self.calculate_power_margin_components(
                        drone_pos=np.asarray(drone_pos_n, dtype=float),
                        station_pos=np.asarray(station_pos, dtype=float),
                        scenario=scenario_no_cache,
                        drone_idx=int(di),
                    )
                    sinr = float(pm.get('SINR_linear', 0.0))
                    r_bps = float(b * np.log2(1.0 + max(sinr, 0.0) / gamma))
                    rates.append(r_bps)
                if not rates:
                    continue
                k = int(np.argmax(np.asarray(rates, dtype=float)))
                bits = float(max(rates[k], 0.0)) * float(dt)
                station_bits[k] += bits
                total_bits += bits
                comm_time += float(dt)

            total_energy_j += float(pc * comm_time)

        ee_bpj = float(total_bits / max(total_energy_j, 1e-12))
        sum_rate_bps = float(total_bits / max(float(len(t)) * dt, 1e-12))
        sum_propulsion_w = float(total_energy_j / max(float(len(t)) * dt, 1e-12))
        out = {
            'sum_rate_bps': float(sum_rate_bps),
            'sum_propulsion_w': float(sum_propulsion_w),
            'EE_bpj': float(ee_bpj),
            'tdma_enabled': 1.0,
            'tdma_total_bits': float(total_bits),
            'tdma_total_energy_j': float(total_energy_j),
            'tdma_station_bits': station_bits.tolist(),
            'traj_dt_s': float(dt),
            'traj_steps': int(len(t)),
        }
        return out

    def calculate_outline_total_score(self, scenario: Dict) -> Dict[str, float]:
        """
        大纲式总评分（E3）：把“通信劣化 + 吞吐违约 + 能效劣化 + 约束罚”融合成 D_total（越大越差）。

        约定：
        - D_comm：沿用现有 power_score 的 D（或你选择的其他通信劣化口径）
        - D_thr：TDMA 吞吐约束违约（可选）
        - D_EE：能效劣化（可选）
        """
        pm = self.calculate_power_score_metrics(scenario)
        d_comm = float(pm.get('D', 0.0))

        ee = self.calculate_energy_efficiency_metrics(scenario)
        ee_bpj = float(ee.get('EE_bpj', 0.0))

        d_thr = 0.0
        if bool(self.tdma_enabled):
            # 目标吞吐支持两种口径：bits 或 bps
            q_bits = self.user_params.get('tdma_Qk_bits', None)
            q_bps = self.user_params.get('tdma_Qk_bps', None)
            station_bits = ee.get('tdma_station_bits', None)
            dt = float(ee.get('traj_dt_s', self.user_params.get('traj_dt_s', 1.0)))
            steps = int(ee.get('traj_steps', max(1, int(round(float(self.user_params.get('traj_total_time_s', 20.0)) / max(dt, 1e-6))))))
            t_total = float(max(steps, 1) * dt)

            if isinstance(station_bits, list) and station_bits:
                station_bits_arr = np.asarray(station_bits, dtype=float)
                if q_bits is None and q_bps is not None:
                    if isinstance(q_bps, (int, float)):
                        q_bits_arr = np.ones_like(station_bits_arr) * float(q_bps) * t_total
                    else:
                        q_bits_arr = np.asarray(q_bps, dtype=float) * t_total
                elif q_bits is not None:
                    if isinstance(q_bits, (int, float)):
                        q_bits_arr = np.ones_like(station_bits_arr) * float(q_bits)
                    else:
                        q_bits_arr = np.asarray(q_bits, dtype=float)
                else:
                    q_bits_arr = None

                if q_bits_arr is not None and q_bits_arr.size == station_bits_arr.size:
                    denom = np.maximum(q_bits_arr, 1e-12)
                    d_thr = float(np.sum(np.maximum(0.0, q_bits_arr - station_bits_arr) / denom))

        ee_target = self.user_params.get('ee_target_bpj', None)
        if ee_target is not None:
            ee_target = max(float(ee_target), 1e-12)
            d_ee = float(np.clip(max(0.0, (ee_target - ee_bpj) / ee_target), 0.0, 1.0))
        else:
            ee_ref = max(float(self.user_params.get('ee_ref_bpj', 1e5)), 1e-12)
            d_ee = float(1.0 / (1.0 + ee_bpj / ee_ref))

        d_pen = 0.0
        w = dict(getattr(self, 'outline_weights', {}))
        d_total = float(
            float(w.get('w_comm', 1.0)) * d_comm +
            float(w.get('w_thr', 0.0)) * d_thr +
            float(w.get('w_ee', 0.0)) * d_ee +
            float(w.get('w_penalty', 0.0)) * d_pen
        )
        return {
            'D_total': float(d_total),
            'D_comm': float(d_comm),
            'D_thr': float(d_thr),
            'D_EE': float(d_ee),
            'D_penalty': float(d_pen),
            'EE_bpj': float(ee_bpj),
        }

    def _get_aerial_interferers(self, scenario: Dict) -> List['InterferenceSource']:
        """
        “空中干扰项” I_aer 的来源入口（Zeng 2019 的 I_aer(Q) 结构）。

        重要澄清：
        - UAV↔UAV 通信在本项目中按“有用信道/有用链路”理解（见 u2u_enabled），不应作为 I_aer 干扰项。
        - I_aer 仅在你显式提供“空中干扰机/同频空中干扰源”集合时启用。

        约定：用户可在 scenario 里提供
        - aerial_interferers: List[InterferenceSource]

        兼容旧键：aerial_interference_sources（将其视为 aerial_interferers）。
        """
        lst = scenario.get('aerial_interferers', None)
        if isinstance(lst, list):
            return lst
        legacy = scenario.get('aerial_interference_sources', None)
        if isinstance(legacy, list):
            return legacy
        return []

    def calculate_aerial_interference_power_mw(self,
                                               rx_pos: np.ndarray,
                                               aerial_interferers: List['InterferenceSource'],
                                               scenario: Dict,
                                               rx_freq_hz: float,
                                               rx_bw_hz: float,
                                               rx_idx: Optional[int] = None) -> float:
        """
        计算空中干扰功率 I_aer（mW）：按与 S(.) 同构的功率模板计算每个空中干扰源的到达功率并线性累加。

        注意：这里默认按“空对空”/“空对点”的 mean path loss（LoS/NLoS）计算；
        若你后续要更严格区分 U2U/ATG，可在此处替换 path loss 模块。
        """
        if not aerial_interferers:
            return 0.0

        p_rx = np.asarray(rx_pos, dtype=float)
        buildings_list = scenario.get('buildings', [])
        total_mw = 0.0
        for j, src in enumerate(aerial_interferers):
            tx_pos = np.array([float(getattr(src, 'x')), float(getattr(src, 'y')), float(getattr(src, 'height'))], dtype=float)
            tx_power_dbm = float(getattr(src, 'power'))
            tx_freq = float(getattr(src, 'frequency', rx_freq_hz))

            coupling = self._spectral_coupling_factor(
                tx_freq_hz=tx_freq,
                rx_freq_hz=float(rx_freq_hz),
                rx_bw_hz=float(rx_bw_hz),
                interferer_type_id=int(getattr(src, 'type_id', -1)),
            )
            if coupling <= 0.0:
                continue

            # 可复现的“干扰衰落”
            fade = 1.0
            if getattr(self, 'interference_fading_enabled', False):
                u = self._stable_unit_float(
                    17.0,
                    float(rx_idx if rx_idx is not None else 0),
                    float(j) + 1.0,
                    float(tx_power_dbm),
                    float(tx_pos[0]), float(tx_pos[1]), float(tx_pos[2]),
                    float(p_rx[0]), float(p_rx[1]), float(p_rx[2]),
                )
                u = float(np.clip(u, 1e-12, 1.0 - 1e-12))
                fade = float(-math.log(1.0 - u))  # Exp(1)

            act = float(getattr(src, 'activity', 1.0))
            rx_dbm, _ = self.calculate_rx_power_dbm(
                tx_power_dbm=tx_power_dbm,
                tx_pos=tx_pos,
                rx_pos=p_rx,
                freq_hz=float(tx_freq),
                buildings=buildings_list,
                force_nlos=False,
                ignore_building_idx=None,
                extra_loss_db=0.0,
                fading_gain_linear=float(fade),
                duty_cycle_activity=float(act),
            )
            total_mw += self._dbm_to_mw(rx_dbm) * float(coupling)

        return float(total_mw)

    def compute_sinr(self,
                     drone_pos: np.ndarray,
                     station_pos: np.ndarray,
                     scenario: Dict,
                     drone_idx: Optional[int] = None) -> Dict[str, float]:
        """SINR 统一接口：内部复用功率域模板，返回 SINR/Outage/速率等。"""
        return self.calculate_power_margin_components(drone_pos, station_pos, scenario, drone_idx=drone_idx)

    def compute_sinr_zeng(self,
                          drone_pos: np.ndarray,
                          station_pos: np.ndarray,
                          scenario: Dict,
                          drone_idx: Optional[int] = None,
                          include_aerial_interferers: bool = True) -> Dict[str, float]:
        """
        Zeng 2019 “论文同款” SINR 接口（功率域实现）：
        γ = S / (I_ter + I_aer + σ^2)

        - S：沿用你现有的接收功率/LoS/NLoS/遮挡模型（不推翻）
        - I_ter：默认使用 scenario['interference_sources']（已预计算可复用）
        - I_aer：可选，来自 scenario['aerial_interferers']（空中干扰机/同频空中干扰源）；UAV↔UAV 通信不计入此项
        """
        pm = self.calculate_power_margin_components(drone_pos, station_pos, scenario, drone_idx=drone_idx)

        i_ter_mw = float(pm.get('I_y_mw', 0.0))
        i_aer_mw = 0.0
        if include_aerial_interferers:
            aer_sources = self._get_aerial_interferers(scenario)
            i_aer_mw = float(self.calculate_aerial_interference_power_mw(
                rx_pos=np.asarray(drone_pos, dtype=float),
                aerial_interferers=aer_sources,
                scenario=scenario,
                rx_freq_hz=float(self.communication_freq),
                rx_bw_hz=float(getattr(self, 'bandwidth_hz', 20e6)),
                rx_idx=drone_idx,
            ))

        i_total_mw = float(max(i_ter_mw, 0.0) + max(i_aer_mw, 0.0))
        ny_mw = float(pm.get('N_y_mw', self._noise_power_mw()))
        pin_mw = float(i_total_mw + max(ny_mw, 0.0))
        pin_dbm = float(self._mw_to_dbm(pin_mw))
        n_dbm = float(self._mw_to_dbm(max(ny_mw, 0.0)))
        delta_in_db = float(pin_dbm - n_dbm)

        prx_dbm = float(pm.get('P_rx_dbm', -300.0))
        m_db = float(prx_dbm - pin_dbm)
        sinr_lin = float(10.0 ** (m_db / 10.0))
        b = max(float(getattr(self, 'bandwidth_hz', 1.0)), 1.0)
        rate_bps = float(b * np.log2(1.0 + sinr_lin))
        outage = 1.0 if m_db < float(getattr(self, 'margin_threshold_db', 0.0)) else 0.0

        out = dict(pm)
        out.update({
            'I_ter_mw': float(i_ter_mw),
            'I_aer_mw': float(i_aer_mw),
            'I_total_mw': float(i_total_mw),
            'I_y_mw': float(i_total_mw),
            'I_y_dbm': float(self._mw_to_dbm(max(i_total_mw, 0.0))),
            'P_IN_dbm': float(pin_dbm),
            'Delta_IN_db': float(delta_in_db),
            'M_db': float(m_db),
            'SINR_linear': float(sinr_lin),
            'R_bps': float(rate_bps),
            'Outage': float(outage),
        })
        return out

    def calculate_link_degradation(self,
                                   drone_pos: np.ndarray,
                                   station_pos: np.ndarray,
                                   scenario: Dict,
                                   drone_speed_mps: Optional[float] = None,
                                   drone_idx: Optional[int] = None) -> float:
        comm_deg = self.calculate_comm_degradation(drone_pos, station_pos, scenario, drone_idx=drone_idx)

        if drone_speed_mps is None:
            speeds = scenario.get('drone_speeds', [])
            drone_speed_mps = float(np.mean(speeds)) if len(speeds) else 0.0

        speed_eff_deg = self.calculate_speed_energy_efficiency_degradation(drone_speed_mps)

        weight_sum = sum(self.metric_weights.values())
        total = (
            self.metric_weights['power_intensity_change'] * comm_deg +
            self.metric_weights['speed_energy_efficiency'] * speed_eff_deg
        ) / max(weight_sum, 1e-12)

        return float(np.clip(total, 0.0, 1.0))

    def analyze_scenario_metrics(self, scenario: Dict) -> Dict:
        """为可视化/报告生成一些汇总指标（不参与 GA）。"""
        drones = np.asarray(scenario.get('drone_positions', np.zeros((0, 3))), dtype=float)
        stations = np.asarray(scenario.get('station_positions', np.zeros((0, 3))), dtype=float)
        if drones.size == 0 or stations.size == 0:
            return {
                'avg_comm_deg': 0.0,
                'avg_speed_deg': 0.0,
                'avg_total_deg': 0.0,
                'avg_d_link_db': 0.0,
                'avg_interference_dbm': -np.inf,
                'avg_cdi': 0.0,
                'avg_margin_db': 0.0,
                'outage_prob': 0.0,
                'avg_rate_mbps': 0.0,
                'avg_sinr_db': -np.inf,
                'model': str(getattr(self, 'degradation_model', 'margin')),
            }

        comm_degs = []
        total_degs = []
        int_mws = []
        margins = []
        outages = []
        rates_mbps = []
        sinr_dbs = []
        prx_dbms = []
        pin_dbms = []
        delta_in_dbs = []
        iy_dbms = []
        i_aer_dbms = []
        u2u_comm_degs = []

        for drone_idx in range(len(drones)):
            for station_idx in range(len(stations)):
                use_zeng = bool(self.user_params.get('sinr_zeng_enabled', False))
                if use_zeng:
                    pm = self.compute_sinr_zeng(
                        drones[drone_idx],
                        stations[station_idx],
                        scenario,
                        drone_idx=int(drone_idx),
                        include_aerial_interferers=True,
                    )
                else:
                    pm = self.calculate_power_margin_components(
                        drones[drone_idx],
                        stations[station_idx],
                        scenario,
                        drone_idx=int(drone_idx),
                    )
                margins.append(float(pm.get('M_db', 0.0)))
                outages.append(float(pm.get('Outage', 0.0)))
                rates_mbps.append(float(pm.get('R_bps', 0.0)) / 1e6)
                sinr_dbs.append(float(10.0 * np.log10(max(float(pm.get('SINR_linear', 0.0)), 1e-15))))
                prx_dbms.append(float(pm.get('P_rx_dbm', 0.0)))
                pin_dbms.append(float(pm.get('P_IN_dbm', 0.0)))
                delta_in_dbs.append(float(pm.get('Delta_IN_db', 0.0)))
                iy_dbms.append(float(pm.get('I_y_dbm', -np.inf)))
                if 'I_aer_mw' in pm:
                    i_aer_dbms.append(float(self._mw_to_dbm(max(float(pm.get('I_aer_mw', 0.0)), 0.0))))

                # 单独记录 U2U 的“有用信道”劣化，便于解释（不进入 I_aer）
                u2u_d = self._u2u_comm_degradation_for_drone(scenario, int(drone_idx))
                if u2u_d is not None:
                    u2u_comm_degs.append(float(u2u_d))

                # CDI 指标已停用：只记录干扰强度用于可视化/统计
                i_mw = float(self.calculate_interference_power_mw(
                    drone_pos=np.asarray(drones[drone_idx], dtype=float),
                    interference_sources=scenario.get('interference_sources', []),
                    buildings=scenario.get('buildings', []),
                    drone_idx=int(drone_idx),
                ))
                int_mws.append(float(i_mw))
                comm_degs.append(float(self.calculate_comm_degradation(
                    drones[drone_idx],
                    stations[station_idx],
                    scenario,
                    drone_idx=int(drone_idx),
                )))
                total_degs.append(float(self.calculate_link_degradation(
                    drones[drone_idx],
                    stations[station_idx],
                    scenario,
                    drone_speed_mps=float(scenario.get('drone_speeds', [0.0] * len(drones))[drone_idx]),
                    drone_idx=int(drone_idx),
                )))

        speed_degs = []
        for v in scenario.get('drone_speeds', []):
            speed_degs.append(float(self.calculate_speed_energy_efficiency_degradation(float(v))))

        avg_i_mw = float(np.mean(int_mws)) if int_mws else 0.0
        avg_i_dbm = float(self._mw_to_dbm(avg_i_mw)) if np.isfinite(avg_i_mw) else -np.inf
        ee = self.calculate_energy_efficiency_metrics(scenario)
        outline = {}
        try:
            outline = self.calculate_outline_total_score(scenario)
        except Exception:
            outline = {}
        metrics = {
            'avg_comm_deg': float(np.mean(comm_degs)) if comm_degs else 0.0,
            'avg_speed_deg': float(np.mean(speed_degs)) if speed_degs else 0.0,
            'avg_total_deg': float(np.mean(total_degs)) if total_degs else 0.0,
            'avg_d_link_db': 0.0,
            'avg_interference_dbm': avg_i_dbm,
            'avg_cdi': 0.0,
            'avg_margin_db': float(np.mean(margins)) if margins else 0.0,
            'outage_prob': float(np.mean(outages)) if outages else 0.0,
            'avg_rate_mbps': float(np.mean(rates_mbps)) if rates_mbps else 0.0,
            'avg_sinr_db': float(np.mean(sinr_dbs)) if sinr_dbs else -np.inf,
            'EE_bpj': float(ee.get('EE_bpj', 0.0)),
            'sum_rate_bps': float(ee.get('sum_rate_bps', 0.0)),
            'sum_propulsion_w': float(ee.get('sum_propulsion_w', 0.0)),
            'tdma_enabled': float(ee.get('tdma_enabled', 0.0)) if isinstance(ee, dict) else 0.0,
            'tdma_total_bits': float(ee.get('tdma_total_bits', 0.0)) if isinstance(ee, dict) else 0.0,
            'tdma_total_energy_j': float(ee.get('tdma_total_energy_j', 0.0)) if isinstance(ee, dict) else 0.0,
            'avg_prx_dbm': float(np.mean(prx_dbms)) if prx_dbms else 0.0,
            'avg_pin_dbm': float(np.mean(pin_dbms)) if pin_dbms else 0.0,
            'avg_delta_in_db': float(np.mean(delta_in_dbs)) if delta_in_dbs else 0.0,
            'avg_iy_dbm': float(np.mean([v for v in iy_dbms if np.isfinite(v)])) if any(np.isfinite(v) for v in iy_dbms) else -np.inf,
            'avg_i_aer_dbm': float(np.mean([v for v in i_aer_dbms if np.isfinite(v)])) if i_aer_dbms and any(np.isfinite(v) for v in i_aer_dbms) else -np.inf,
            'avg_u2u_comm_deg': float(np.mean(u2u_comm_degs)) if u2u_comm_degs else 0.0,
            'model': str(getattr(self, 'degradation_model', 'margin')),
        }
        for k in ('D_total', 'D_comm', 'D_thr', 'D_EE', 'D_penalty'):
            if k in outline:
                metrics[k] = float(outline.get(k, 0.0))
        try:
            ps = self.calculate_power_score_metrics(scenario)
            metrics.update({f'power_{k}': float(v) if isinstance(v, (int, float)) else v for k, v in ps.items()})
        except Exception:
            pass
        return metrics

    def _iter_a2g_links(self, scenario: Dict):
        drones = np.asarray(scenario.get('drone_positions', np.zeros((0, 3))), dtype=float)
        stations = np.asarray(scenario.get('station_positions', np.zeros((0, 3))), dtype=float)
        for drone_idx in range(len(drones)):
            for station_idx in range(len(stations)):
                yield int(drone_idx), int(station_idx), drones[drone_idx], stations[station_idx]

    def _iter_u2u_links(self, scenario: Dict):
        if not getattr(self, 'u2u_enabled', False):
            return
        drones = np.asarray(scenario.get('drone_positions', np.zeros((0, 3))), dtype=float)
        if len(drones) < 2:
            return
        pairing = str(getattr(self, 'u2u_pairing', 'all')).strip().lower()
        if pairing == 'nearest':
            for i in range(len(drones)):
                d2 = np.sum((drones[:, :2] - drones[i, :2]) ** 2, axis=1)
                d2[i] = np.inf
                j = int(np.argmin(d2))
                if np.isfinite(d2[j]):
                    yield int(i), int(j), drones[i], drones[j]
        else:
            for i in range(len(drones)):
                for j in range(i + 1, len(drones)):
                    yield int(i), int(j), drones[i], drones[j]

    def calculate_u2u_power_margin_components(self,
                                              tx_drone_pos: np.ndarray,
                                              rx_drone_pos: np.ndarray,
                                              scenario: Dict,
                                              tx_idx: int,
                                              rx_idx: int) -> Dict[str, float]:
        """U2U：与 A2G 同构的功率预算，只是 tx/rx 都在空中。"""
        buildings_list = scenario.get('buildings', [])
        fade = self._stable_exp_fading_gain_for_link((
            19.0,
            float(tx_idx), float(rx_idx),
            float(tx_drone_pos[0]), float(tx_drone_pos[1]), float(tx_drone_pos[2]),
            float(rx_drone_pos[0]), float(rx_drone_pos[1]), float(rx_drone_pos[2]),
        ))
        prx_dbm, p_los = self.calculate_rx_power_dbm(
            tx_power_dbm=float(self.u2u_tx_power_dbm),
            tx_pos=tx_drone_pos,
            rx_pos=rx_drone_pos,
            freq_hz=float(self.communication_freq),
            buildings=buildings_list,
            force_nlos=False,
            ignore_building_idx=None,
            fading_gain_linear=float(fade),
            duty_cycle_activity=1.0,
        )
        prx_mw = float(self._dbm_to_mw(prx_dbm))
        pl_db, _ = self.calculate_mean_path_loss_db(
            tx_pos=tx_drone_pos, rx_pos=rx_drone_pos, freq_hz=float(self.communication_freq), buildings=buildings_list
        )

        iy_mw: Optional[float] = None
        if 'interference_power_mw_per_drone' in scenario:
            try:
                iy_mw = float(scenario['interference_power_mw_per_drone'][int(rx_idx)])
            except Exception:
                iy_mw = None
        if iy_mw is None:
            iy_mw = float(self.calculate_interference_power_mw(
                drone_pos=np.asarray(rx_drone_pos, dtype=float),
                interference_sources=scenario.get('interference_sources', []),
                buildings=buildings_list,
                drone_idx=rx_idx,
            ))
        ny_mw = float(self._noise_power_mw())
        pin_mw = float(max(iy_mw, 0.0) + max(ny_mw, 0.0))
        pin_dbm = float(self._mw_to_dbm(pin_mw))
        n_dbm = float(self._mw_to_dbm(ny_mw))
        delta_in_db = float(pin_dbm - n_dbm)
        iy_dbm = float(self._mw_to_dbm(max(float(iy_mw), 0.0)))
        m_db = float(prx_dbm - pin_dbm)
        sinr_lin = float(10.0 ** (m_db / 10.0))
        b = max(float(getattr(self, 'bandwidth_hz', 1.0)), 1.0)
        rate_bps = float(b * np.log2(1.0 + sinr_lin))
        outage = 1.0 if m_db < float(getattr(self, 'margin_threshold_db', 0.0)) else 0.0
        return {
            'P_rx_dbm': float(prx_dbm),
            'P_rx_mw': float(prx_mw),
            'I_y_mw': float(iy_mw),
            'I_y_dbm': float(iy_dbm),
            'N_y_mw': float(ny_mw),
            'N_y_dbm': float(n_dbm),
            'P_IN_dbm': float(pin_dbm),
            'Delta_IN_db': float(delta_in_db),
            'M_db': float(m_db),
            'SINR_linear': float(sinr_lin),
            'R_bps': float(rate_bps),
            'Outage': float(outage),
            'PL_db': float(pl_db),
            'P_LoS': float(p_los),
        }

    def calculate_power_score_metrics(self, scenario: Dict) -> Dict[str, float]:
        """
        按大纲的“功率主导”输出三类概率指标：
        - P_succ(θ_p)=P(P_rx>θ_p)
        - P_int(θ_I)=P(I_agg>θ_I)
        - P_PL=P(PL>PL_max)
        并给出最终拼接分数 D = w1(1-P_succ)+w2 P_int + w3 P_PL（越大表示越差）。
        """
        theta_p_dbm = float(getattr(self, 'succ_threshold_dbm', -90.0))
        theta_i_dbm = float(getattr(self, 'int_threshold_dbm', -80.0))
        pl_max_db = float(getattr(self, 'pl_max_db', 125.0))
        w = dict(getattr(self, 'power_score_weights', {}))
        w1 = float(w.get('w1_succ', 1.0))
        w2 = float(w.get('w2_int', 1.0))
        w3 = float(w.get('w3_pl', 1.0))

        prx_hits = 0
        int_hits = 0
        pl_hits = 0
        n_links = 0

        # A2G links (station -> drone)
        for drone_idx, station_idx, drone_pos, station_pos in self._iter_a2g_links(scenario):
            comp = self.calculate_power_margin_components(drone_pos, station_pos, scenario, drone_idx=drone_idx)
            n_links += 1
            if float(comp['P_rx_dbm']) > theta_p_dbm:
                prx_hits += 1
            if float(comp['I_y_dbm']) > theta_i_dbm:
                int_hits += 1
            if float(comp['PL_db']) > pl_max_db:
                pl_hits += 1

        # U2U links (optional)
        if getattr(self, 'u2u_enabled', False):
            for i, j, pi, pj in self._iter_u2u_links(scenario):
                comp_ij = self.calculate_u2u_power_margin_components(pi, pj, scenario, tx_idx=i, rx_idx=j)
                n_links += 1
                if float(comp_ij['P_rx_dbm']) > theta_p_dbm:
                    prx_hits += 1
                if float(comp_ij['I_y_dbm']) > theta_i_dbm:
                    int_hits += 1
                if float(comp_ij['PL_db']) > pl_max_db:
                    pl_hits += 1

        if n_links <= 0:
            return {
                'P_succ': 0.0,
                'P_int': 0.0,
                'P_PL': 0.0,
                'D': 0.0,
                'theta_p_dbm': theta_p_dbm,
                'theta_i_dbm': theta_i_dbm,
                'pl_max_db': pl_max_db,
            }

        p_succ = float(prx_hits / n_links)
        p_int = float(int_hits / n_links)
        p_pl = float(pl_hits / n_links)
        d_score = float(w1 * (1.0 - p_succ) + w2 * p_int + w3 * p_pl)
        return {
            'P_succ': p_succ,
            'P_int': p_int,
            'P_PL': p_pl,
            'D': d_score,
            'theta_p_dbm': theta_p_dbm,
            'theta_i_dbm': theta_i_dbm,
            'pl_max_db': pl_max_db,
        }

    def evalVars(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """评估种群"""
        N = x.shape[0]
        f = np.zeros((N, 1))

        for i in range(N):
            try:
                # 生成场景
                scenario = self.generate_scenario(x[i, :])

                obj_model = str(getattr(self, 'objective_model', 'power_score')).strip().lower()
                if obj_model in ('power_score', 'outline', 'power'):
                    pm = self.calculate_power_score_metrics(scenario)
                    # D 越大表示越差（更“劣化”），与 maxormins=-1 的“最大化”保持一致
                    f[i, 0] = float(pm.get('D', 0.0))
                elif obj_model in ('outline_total', 'd_total', 'total'):
                    out = self.calculate_outline_total_score(scenario)
                    f[i, 0] = float(out.get('D_total', 0.0))
                elif obj_model in ('ee', 'ee_bpj', 'energy', 'energy_efficiency'):
                    ee = self.calculate_energy_efficiency_metrics(scenario)
                    f[i, 0] = float(ee.get('EE_bpj', 0.0))
                elif obj_model in ('ee_log', 'ee_bpj_log', 'energy_log'):
                    ee = self.calculate_energy_efficiency_metrics(scenario)
                    f[i, 0] = float(np.log10(1.0 + float(ee.get('EE_bpj', 0.0))))
                else:
                    # 兼容旧目标：平均劣化（可包含速度项）
                    total_degradation = 0.0
                    link_count = 0
                    for drone_idx in range(self.num_drones):
                        for station_idx in range(self.num_stations):
                            degradation = self.calculate_link_degradation(
                                scenario['drone_positions'][drone_idx],
                                scenario['station_positions'][station_idx],
                                scenario,
                                drone_speed_mps=float(scenario['drone_speeds'][drone_idx]),
                                drone_idx=int(drone_idx)
                            )
                            total_degradation += float(degradation)
                            link_count += 1
                    f[i, 0] = float(total_degradation / link_count) if link_count > 0 else 0.01

            except Exception as e:
                print(f"评估染色体时出错: {e}")
                f[i, 0] = 0.01

        return f, np.zeros((N, 1))


class EnhancedVisualizer:
    """增强版场景可视化类"""

    def __init__(self, scenario: Dict):
        self.scenario = scenario
        # 可视化开关：默认“全量绘制”（用户要求），避免只画子集导致数量不一致
        # 仍保留场景级开关，便于在极端密度下回退到子集显示
        self.viz_plot_all_sources = bool(scenario.get('viz_plot_all_sources', True))
        self.viz_plot_all_buildings = bool(scenario.get('viz_plot_all_buildings', True))
        # 按原来风格：在顶视图标注建筑高度
        self.viz_annotate_building_heights = bool(scenario.get('viz_annotate_building_heights', True))
        self.viz_plot_interference_ranges = bool(
            scenario.get('viz_plot_interference_ranges', not self.viz_plot_all_sources)
        )
        self.colors = {
            'drones': '#FF6B6B',  # 红色 - 无人机（三角形）
            'stations': '#4ECDC4',  # 青色 - 地面站（方形）
            'buildings': '#45B7D1',  # 蓝色 - 建筑物
            'ground': '#DFE6E9',  # 浅灰色 - 地面
            'text': '#2D3436',  # 深灰色 - 文字
            'interference': '#FFD700'  # 金色 - 干扰源（统一为星型）
        }

        # 干扰源颜色映射 - 使用干扰源自身的颜色
        self.interference_colors = {}
        for source in scenario.get('interference_sources', []):
            self.interference_colors[source.name] = source.color

    def _sources_for_plot(self) -> List:
        """获取要绘制的干扰源集合。默认全量绘制；可用 viz_plot_all_sources 回退到子集。"""
        if self.viz_plot_all_sources:
            sources = self.scenario.get('interference_sources', [])
        else:
            sources = self.scenario.get('interference_sources_viz', None)
            if sources is None:
                sources = self.scenario.get('interference_sources', [])
        return list(sources) if sources else []

    @staticmethod
    def _count_sources_by_name(sources: List) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in sources or []:
            name = getattr(s, 'name', 'unknown')
            counts[name] = counts.get(name, 0) + 1
        return counts

    @staticmethod
    def _count_ground_building(sources: List) -> Tuple[int, int]:
        ground = 0
        building = 0
        for s in sources or []:
            if getattr(s, 'is_ground', False):
                ground += 1
            else:
                building += 1
        return ground, building

    def _format_source_summary(self, sources_plot: List, sources_all: List) -> str:
        shown_total = len(sources_plot or [])
        total_total = len(sources_all or [])
        shown_ground, shown_building = self._count_ground_building(sources_plot)
        total_ground, total_building = self._count_ground_building(sources_all)
        return (
            f"干扰源: 显示{shown_total}/总{total_total}\n"
            f"  建筑: {shown_building}/{total_building}\n"
            f"  地面: {shown_ground}/{total_ground}"
        )

    def _buildings_for_plot(self, limit: int = 60) -> List['Building']:
        """获取要绘制的建筑集合。默认全量绘制；否则确保与可视化干扰源关联的建筑一定会被画出来。"""
        buildings = list(self.scenario.get('buildings', []))
        if not buildings:
            return []

        if self.viz_plot_all_buildings:
            return buildings

        sources = self._sources_for_plot()
        assoc_idx = []
        for s in sources:
            b_idx = getattr(s, 'building_idx', None)
            b = getattr(s, 'building', None)
            if b_idx is not None:
                assoc_idx.append(int(b_idx))
            elif b is not None:
                try:
                    assoc_idx.append(int(buildings.index(b)))
                except Exception:
                    pass

        assoc_idx = sorted(set([i for i in assoc_idx if 0 <= i < len(buildings)]))
        assoc_buildings = [buildings[i] for i in assoc_idx]

        limit = int(max(limit, len(assoc_buildings)))
        if len(buildings) <= limit:
            return buildings

        remaining = [b for i, b in enumerate(buildings) if i not in set(assoc_idx)]
        remaining.sort(key=lambda bb: float(getattr(bb, 'height', 0.0)), reverse=True)
        return assoc_buildings + remaining[: max(0, limit - len(assoc_buildings))]

    def create_individual_plots(self, prefix: str = "") -> List[Tuple[str, str]]:
        """创建单独的图表"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"{prefix}_{timestamp}" if prefix else timestamp

        plots_info = []

        try:
            # 1. 3D场景图
            fig1 = plt.figure(figsize=(14, 10))
            ax1 = fig1.add_subplot(111, projection='3d')
            self._plot_3d_scene(ax1)
            plt.tight_layout()
            filename1 = f"output/3d_scene_{prefix}.png"
            plt.savefig(filename1, dpi=150, bbox_inches='tight')
            plt.close(fig1)
            plots_info.append(("3D场景图", filename1))

            # 2. 顶视图
            fig2, ax2 = plt.subplots(figsize=(12, 10))
            self._plot_top_view(ax2)
            plt.tight_layout()
            filename2 = f"output/top_view_{prefix}.png"
            plt.savefig(filename2, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            plots_info.append(("顶视图", filename2))

            # 3. 侧视图
            fig3, ax3 = plt.subplots(figsize=(12, 10))
            self._plot_side_view(ax3)
            plt.tight_layout()
            filename3 = f"output/side_view_{prefix}.png"
            plt.savefig(filename3, dpi=150, bbox_inches='tight')
            plt.close(fig3)
            plots_info.append(("侧视图", filename3))

            # 4. 干扰源分布
            fig4, ax4 = plt.subplots(figsize=(12, 8))
            self._plot_interference_distribution(ax4)
            plt.tight_layout()
            filename4 = f"output/interference_dist_{prefix}.png"
            plt.savefig(filename4, dpi=150, bbox_inches='tight')
            plt.close(fig4)
            plots_info.append(("干扰源分布", filename4))

            # 5. 高度分析
            fig5, ax5 = plt.subplots(figsize=(12, 8))
            self._plot_height_analysis(ax5)
            plt.tight_layout()
            filename5 = f"output/height_analysis_{prefix}.png"
            plt.savefig(filename5, dpi=150, bbox_inches='tight')
            plt.close(fig5)
            plots_info.append(("高度分析", filename5))

            # 6. 通信链路分析
            fig6, ax6 = plt.subplots(figsize=(12, 8))
            self._plot_communication_analysis(ax6)
            plt.tight_layout()
            filename6 = f"output/communication_analysis_{prefix}.png"
            plt.savefig(filename6, dpi=150, bbox_inches='tight')
            plt.close(fig6)
            plots_info.append(("通信链路分析", filename6))

            # 7. 场景参数图
            fig7, ax7 = plt.subplots(figsize=(12, 8))
            self._plot_scenario_parameters(ax7)
            plt.tight_layout()
            filename7 = f"output/scenario_params_{prefix}.png"
            plt.savefig(filename7, dpi=150, bbox_inches='tight')
            plt.close(fig7)
            plots_info.append(("场景参数", filename7))

            # 8. 干扰源位置类型分析
            fig8, ax8 = plt.subplots(figsize=(12, 8))
            self._plot_interference_location_type(ax8)
            plt.tight_layout()
            filename8 = f"output/interference_location_{prefix}.png"
            plt.savefig(filename8, dpi=150, bbox_inches='tight')
            plt.close(fig8)
            plots_info.append(("干扰源位置类型", filename8))

        except Exception as e:
            print(f"生成图表时出错: {e}")
            import traceback
            traceback.print_exc()

        return plots_info

    def create_comprehensive_report(self, prefix: str = "") -> str:
        """创建综合报告图"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"output/comprehensive_report_{prefix}_{timestamp}.png"

        fig = plt.figure(figsize=(20, 16))

        # 1. 3D场景图
        ax1 = fig.add_subplot(331, projection='3d')
        self._plot_3d_scene(ax1)

        # 2. 顶视图
        ax2 = fig.add_subplot(332)
        self._plot_top_view(ax2)

        # 3. 侧视图
        ax3 = fig.add_subplot(333)
        self._plot_side_view(ax3)

        # 4. 干扰源分析
        ax4 = fig.add_subplot(334)
        self._plot_interference_analysis(ax4)

        # 5. 高度分析
        ax5 = fig.add_subplot(335)
        self._plot_detailed_height_analysis(ax5)

        # 6. 通信链路分析（极坐标雷达图）
        ax6 = fig.add_subplot(336, polar=True)
        self._plot_detailed_communication_analysis(ax6)

        # 7. 场景参数
        ax7 = fig.add_subplot(337)
        self._plot_detailed_scenario_parameters(ax7)

        # 8. 干扰源详情
        ax8 = fig.add_subplot(338)
        self._plot_interference_details(ax8)

        # 9. 干扰源位置类型
        ax9 = fig.add_subplot(339)
        self._plot_interference_location_type(ax9)

        plt.suptitle('无人机通信场景综合报告', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()

        return filename

    def _plot_3d_scene(self, ax):
        """绘制3D场景"""
        area_size = self.scenario['area_size']

        # 绘制地面
        xx, yy = np.meshgrid([0, area_size], [0, area_size])
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.1, color=self.colors['ground'])

        # 绘制无人机 - 使用三角形标记
        drones = self.scenario['drone_positions']
        if len(drones) > 0:
            ax.scatter(drones[:, 0], drones[:, 1], drones[:, 2],
                       c=self.colors['drones'], s=150, marker='^',
                       label=f'无人机 ({len(drones)}个)', depthshade=True)

            # 标注无人机高度和速度
            for i, drone in enumerate(drones):
                speed = self.scenario['drone_speeds'][i] if i < len(self.scenario['drone_speeds']) else 0
                ax.text(drone[0], drone[1], drone[2] + 10,
                        f'U{i}:{drone[2]:.0f}m\n{speed:.1f}m/s',
                        fontsize=8, ha='center')

        # 绘制地面站 - 使用方形标记
        stations = self.scenario['station_positions']
        if len(stations) > 0:
            ax.scatter(stations[:, 0], stations[:, 1], stations[:, 2],
                       c=self.colors['stations'], s=120, marker='s',
                       label=f'地面站 ({len(stations)}个)', depthshade=True)

            for i, station in enumerate(stations):
                ax.text(station[0], station[1], station[2] + 5,
                        f'S{i}', fontsize=8, ha='center')

        # 绘制建筑物
        buildings = self._buildings_for_plot(limit=60)
        for building in buildings:
            # 绘制建筑物立方体
            vertices = [
                [building.x - building.width / 2, building.y - building.length / 2, 0],
                [building.x + building.width / 2, building.y - building.length / 2, 0],
                [building.x + building.width / 2, building.y + building.length / 2, 0],
                [building.x - building.width / 2, building.y + building.length / 2, 0],
                [building.x - building.width / 2, building.y - building.length / 2, building.height],
                [building.x + building.width / 2, building.y - building.length / 2, building.height],
                [building.x + building.width / 2, building.y + building.length / 2, building.height],
                [building.x - building.width / 2, building.y + building.length / 2, building.height]
            ]

            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # 侧面
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # 侧面
            ]

            ax.add_collection3d(Poly3DCollection(faces, facecolors=building.color,
                                                 alpha=0.6, linewidths=0.5, edgecolors='k'))

        # 绘制干扰源 - 优先用可视化子集，避免“建筑未画出导致看起来悬空”
        sources_all = list(self.scenario.get('interference_sources', []) or [])
        sources_plot = self._sources_for_plot()
        total_by_name = self._count_sources_by_name(sources_all)
        shown_by_name = self._count_sources_by_name(sources_plot)

        grouped: Dict[str, List] = {}
        for s in sources_plot:
            grouped.setdefault(getattr(s, 'name', 'unknown'), []).append(s)

        for name, group in grouped.items():
            xs = [float(s.x) for s in group]
            ys = [float(s.y) for s in group]
            zs = [float(s.height) for s in group]
            sizes = [80.0 + float(getattr(s, 'power', 0.0)) * 2.0 for s in group]
            colors = [getattr(s, 'color', self.colors['interference']) for s in group]
            label = f"{name} ({shown_by_name.get(name, 0)}/{total_by_name.get(name, 0)})"
            ax.scatter(xs, ys, zs, c=colors, s=sizes, marker='*', label=label, depthshade=True)

        # 绘制干扰范围（简化表示）：仍按“显示子集”逐个画，避免过度拥挤
        if self.viz_plot_interference_ranges and len(sources_plot) <= 200:
            for source in sources_plot:
                theta = np.linspace(0, 2 * np.pi, 20)
                r = float(getattr(source, 'coverage_radius', 0.0)) * 0.1
                if r > 0:
                    color = getattr(source, 'color', self.colors['interference'])
                    x_circle = source.x + r * np.cos(theta)
                    y_circle = source.y + r * np.sin(theta)
                    z_circle = np.full_like(x_circle, source.height + 2)
                    ax.plot(x_circle, y_circle, z_circle, color=color, alpha=0.25, linewidth=0.8)

        # 设置坐标轴
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('高度 (m)')
        ax.set_title('3D场景视图')
        ax.set_xlim(0, area_size)
        ax.set_ylim(0, area_size)
        max_height = max(self.scenario['drone_height_max'], self.scenario['building_height_max'])
        ax.set_zlim(0, max_height + 50)

        # 显示“真实总数 vs 当前展示数量”（解决：生成数量未展现）
        ax.text2D(0.02, 0.98, self._format_source_summary(sources_plot, sources_all),
                  transform=ax.transAxes, va='top', ha='left', fontsize=9,
                  bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

        # 简化图例
        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h
        if unique:
            ax.legend(unique.values(), unique.keys(), loc='upper right', fontsize=8)

        ax.view_init(elev=35, azim=45)
        ax.grid(True, alpha=0.3)

    def _plot_top_view(self, ax):
        """绘制顶视图"""
        area_size = self.scenario['area_size']

        # 绘制无人机 - 三角形标记
        drones = self.scenario['drone_positions']
        if len(drones) > 0:
            ax.scatter(drones[:, 0], drones[:, 1],
                       c=self.colors['drones'], s=100, marker='^', label='无人机')

        # 绘制地面站 - 方形标记
        stations = self.scenario['station_positions']
        if len(stations) > 0:
            ax.scatter(stations[:, 0], stations[:, 1],
                       c=self.colors['stations'], s=100, marker='s', label='地面站')

        # 绘制建筑物
        buildings = self._buildings_for_plot(limit=80)
        for building in buildings:
            rect = Rectangle((building.x - building.width / 2, building.y - building.length / 2),
                             building.width, building.length,
                              facecolor=building.color, alpha=0.6, edgecolor='k', linewidth=0.5)
            ax.add_patch(rect)

            # 标注建筑物高度（全量绘制时默认关闭，避免遮挡）
            if self.viz_annotate_building_heights:
                ax.text(building.x, building.y, f'{building.height:.0f}m',
                        fontsize=7, ha='center', va='center')

        # 绘制干扰源 - 优先用可视化子集
        sources_all = list(self.scenario.get('interference_sources', []) or [])
        sources_plot = self._sources_for_plot()
        for source in sources_plot:
            color = getattr(source, 'color', self.colors['interference'])
            ax.scatter(source.x, source.y, c=color, s=120, marker='*')

            freq_ghz = float(getattr(source, 'frequency', 0.0)) / 1e9
            location_type = "地面" if getattr(source, 'is_ground', False) else "建筑物/空中"
            ax.text(source.x, source.y + 5, f'{source.name}\n{location_type}\n{freq_ghz:.1f}GHz',
                    fontsize=6, ha='center', va='bottom')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('顶视图 (XY平面)')
        ax.set_xlim(0, area_size)
        ax.set_ylim(0, area_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        ax.text(0.02, 0.98, self._format_source_summary(sources_plot, sources_all),
                transform=ax.transAxes, va='top', ha='left', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

    def _plot_side_view(self, ax):
        """绘制侧视图"""
        area_size = self.scenario['area_size']

        # 绘制无人机 - 三角形标记
        drones = self.scenario['drone_positions']
        if len(drones) > 0:
            ax.scatter(drones[:, 0], drones[:, 2],
                       c=self.colors['drones'], s=100, marker='^', label='无人机')

        # 绘制地面站 - 方形标记
        stations = self.scenario['station_positions']
        if len(stations) > 0:
            ax.scatter(stations[:, 0], stations[:, 2],
                       c=self.colors['stations'], s=100, marker='s', label='地面站')

        # 绘制建筑物
        buildings = self._buildings_for_plot(limit=60)
        for building in buildings:
            rect = Rectangle((building.x - building.width / 2, 0),
                             building.width, building.height,
                             facecolor=building.color, alpha=0.6, edgecolor='k', linewidth=0.5)
            ax.add_patch(rect)

        # 绘制干扰源 - 优先用可视化子集
        sources_all = list(self.scenario.get('interference_sources', []) or [])
        sources_plot = self._sources_for_plot()
        for source in sources_plot:
            color = getattr(source, 'color', self.colors['interference'])
            ax.scatter(source.x, source.height, c=color, s=100, marker='*')

        # 绘制高度参考线
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(area_size * 0.8, 110, '低空 (<100m)', fontsize=8, color='gray')
        ax.axhline(y=300, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(area_size * 0.8, 310, '高空 (>300m)', fontsize=8, color='gray')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('高度 (m)')
        ax.set_title('侧视图 (XZ平面)')
        ax.set_xlim(0, area_size)
        max_height = max(self.scenario['drone_height_max'], self.scenario['building_height_max'])
        ax.set_ylim(0, max_height + 50)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        ax.text(0.02, 0.98, self._format_source_summary(sources_plot, sources_all),
                transform=ax.transAxes, va='top', ha='left', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

    def _plot_interference_distribution(self, ax):
        """绘制干扰源分布"""
        if 'interference_sources' not in self.scenario or not self.scenario['interference_sources']:
            ax.text(0.5, 0.5, '无干扰源', ha='center', va='center', fontsize=12)
            ax.set_title('干扰源分布')
            return

        sources = self.scenario['interference_sources']

        # 按类型和位置类型统计
        type_counts = {}
        ground_counts = {}
        building_counts = {}

        for source in sources:
            type_name = source.name
            if type_name not in type_counts:
                type_counts[type_name] = 0
                ground_counts[type_name] = 0
                building_counts[type_name] = 0
            type_counts[type_name] += 1
            if source.is_ground:
                ground_counts[type_name] += 1
            else:
                building_counts[type_name] += 1

        # 绘制分组条形图
        types = list(type_counts.keys())
        x = np.arange(len(types))
        width = 0.35

        # 计算地面和建筑物干扰源数量
        ground_values = [ground_counts[t] for t in types]
        building_values = [building_counts[t] for t in types]

        # 使用干扰源颜色
        colors = [self.interference_colors.get(t, '#DDA0DD') for t in types]

        bars1 = ax.bar(x - width / 2, ground_values, width, label='地面干扰源',
                       color=colors, alpha=0.8, edgecolor='k')
        bars2 = ax.bar(x + width / 2, building_values, width, label='建筑物干扰源',
                       color=colors, alpha=0.6, edgecolor='k', hatch='//')

        ax.set_xlabel('干扰源类型')
        ax.set_ylabel('数量')
        ax.set_title('干扰源类型与位置分布')
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 标注数值
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            '%d' % int(height), ha='center', va='bottom', fontsize=8)

    def _plot_interference_location_type(self, ax):
        """绘制干扰源位置类型分布"""
        if 'interference_sources' not in self.scenario or not self.scenario['interference_sources']:
            ax.text(0.5, 0.5, '无干扰源', ha='center', va='center', fontsize=12)
            ax.set_title('干扰源位置类型')
            return

        sources = self.scenario['interference_sources']

        # 统计位置类型：地面 / 屋顶(室外) / 室内(楼层)
        ground_count = sum(1 for s in sources if getattr(s, 'is_ground', False))
        indoor_count = sum(1 for s in sources if getattr(s, 'is_indoor', False))
        rooftop_or_outdoor_count = len(sources) - ground_count - indoor_count

        # 绘制饼图（若没有室内源则保持两类）
        if indoor_count > 0:
            labels = ['地面', '屋顶/室外楼层', '室内(楼层)']
            sizes = [ground_count, rooftop_or_outdoor_count, indoor_count]
            colors = ['#FF6B6B', '#45B7D1', '#96CEB4']
        else:
            labels = ['地面干扰源', '建筑物干扰源']
            sizes = [ground_count, len(sources) - ground_count]
            colors = ['#FF6B6B', '#45B7D1']

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90)

        ax.set_title('干扰源位置类型分布')

        # 美化文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

    def _plot_height_analysis(self, ax):
        """绘制高度分析"""
        drones = self.scenario['drone_positions']
        buildings = self.scenario['buildings']
        sources = self.scenario.get('interference_sources', [])

        # 收集高度数据
        drone_heights = drones[:, 2] if len(drones) > 0 else np.array([], dtype=float)
        building_heights = [b.height for b in buildings] if buildings else []

        # 收集干扰源高度数据
        ground_interference_heights = [s.height for s in sources if getattr(s, 'is_ground', False)]
        building_interference_heights = [
            s.height for s in sources
            if (not getattr(s, 'is_ground', False)) and (not getattr(s, 'is_indoor', False))
        ]
        indoor_interference_heights = [s.height for s in sources if getattr(s, 'is_indoor', False)]

        # 创建子图
        if drone_heights.size > 0:
            # 绘制所有高度数据的直方图
            all_heights = []
            all_labels = []
            all_colors = []

            if drone_heights.size > 0:
                all_heights.append(drone_heights)
                all_labels.append(f'无人机 (平均:{np.mean(drone_heights):.1f}m)')
                all_colors.append(self.colors['drones'])

            if building_heights:
                all_heights.append(building_heights)
                all_labels.append(f'建筑物 (平均:{np.mean(building_heights):.1f}m)')
                all_colors.append(self.colors['buildings'])

            if ground_interference_heights:
                all_heights.append(ground_interference_heights)
                all_labels.append(f'地面干扰源 (高度:0m)')
                all_colors.append('#FF6B6B')  # 红色

            if building_interference_heights:
                all_heights.append(building_interference_heights)
                all_labels.append(f'建筑物干扰源 (平均:{np.mean(building_interference_heights):.1f}m)')
                all_colors.append('#45B7D1')  # 蓝色

            # 绘制分组直方图
            if indoor_interference_heights:
                all_heights.append(indoor_interference_heights)
                all_labels.append(f'室内(楼层)干扰源 (平均:{np.mean(indoor_interference_heights):.1f}m)')
                all_colors.append('#96CEB4')

            for heights, label, color in zip(all_heights, all_labels, all_colors):
                ax.hist(heights, bins=15, alpha=0.7, label=label, color=color)

            ax.set_xlabel('高度 (m)')
            ax.set_ylabel('数量')
            ax.set_title('高度分布对比')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无高度数据', ha='center', va='center', fontsize=12)
            ax.set_title('高度分析')

    def _plot_communication_analysis(self, ax):
        """绘制通信链路分析"""
        drones = self.scenario['drone_positions']
        stations = self.scenario['station_positions']

        if len(drones) == 0 or len(stations) == 0:
            ax.text(0.5, 0.5, '无通信链路', ha='center', va='center', fontsize=12)
            ax.set_title('通信链路分析')
            return

        area_size = self.scenario['area_size']

        # 计算所有无人机到最近地面站的距离
        distances = []
        for drone in drones:
            min_dist = float('inf')
            for station in stations:
                dist = np.linalg.norm(drone[:2] - station[:2])  # 仅考虑水平距离
                if dist < min_dist:
                    min_dist = dist
            distances.append(min_dist)

        # 绘制距离分布
        ax.hist(distances, bins=min(10, len(distances)), color='#FFEAA7', alpha=0.7, edgecolor='k')
        ax.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2,
                   label=f'平均距离: {np.mean(distances):.1f}m')

        ax.set_xlabel('通信距离 (m)')
        ax.set_ylabel('无人机数量')
        ax.set_title('无人机到最近地面站距离分布')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_scenario_parameters(self, ax):
        """绘制场景参数"""
        ax.axis('off')

        # 基本参数
        text_lines = [
            "场景参数",
            "=" * 40,
            f"区域大小: {self.scenario['area_size']}m × {self.scenario['area_size']}m",
            f"无人机数量: {len(self.scenario['drone_positions'])}",
            f"地面站数量: {len(self.scenario['station_positions'])}",
            f"建筑物数量: {len(self.scenario['buildings'])}",
            f"干扰源数量: {len(self.scenario.get('interference_sources', []))}",
            f"建筑物密度: {self.scenario['building_density']:.2f}",
            f"建筑物最大高度: {self.scenario['building_height_max']:.1f}m",
            f"无人机最大高度: {self.scenario['drone_height_max']:.1f}m",
            f"无人机最大速度: {self.scenario['drone_speed_max']:.1f}m/s",
        ]

        # 无人机信息
        if len(self.scenario['drone_positions']) > 0:
            text_lines.append("\n无人机信息:")
            avg_height = np.mean(self.scenario['drone_positions'][:, 2])
            avg_speed = np.mean(self.scenario['drone_speeds']) if 'drone_speeds' in self.scenario else 0
            text_lines.append(f"  平均高度: {avg_height:.1f}m")
            text_lines.append(f"  平均速度: {avg_speed:.1f}m/s")

        # 干扰源信息
        if self.scenario.get('interference_sources'):
            text_lines.append("\n干扰源信息:")
            ground_count = sum(1 for s in self.scenario['interference_sources'] if s.is_ground)
            building_count = len(self.scenario['interference_sources']) - ground_count
            text_lines.append(f"  地面干扰源: {ground_count}个")
            text_lines.append(f"  建筑物干扰源: {building_count}个")

        # 功率域通信指标（若上层已注入 analysis）
        analysis = self.scenario.get('analysis', {}) if isinstance(self.scenario, dict) else {}
        if analysis:
            text_lines.append("\n通信(功率域)")
            if 'avg_margin_db' in analysis:
                text_lines.append(f"  M(功率裕量): {float(analysis.get('avg_margin_db', 0.0)):.2f} dB")
            if 'outage_prob' in analysis:
                text_lines.append(f"  Outage: {float(analysis.get('outage_prob', 0.0)) * 100.0:.1f}%")
            if 'avg_rate_mbps' in analysis:
                text_lines.append(f"  R: {float(analysis.get('avg_rate_mbps', 0.0)):.2f} Mbps")
            if 'avg_sinr_db' in analysis and np.isfinite(float(analysis.get('avg_sinr_db', -np.inf))):
                text_lines.append(f"  SINR(dB): {float(analysis.get('avg_sinr_db', 0.0)):.2f} dB")

        text = "\n".join(text_lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    def _plot_interference_analysis(self, ax):
        """绘制干扰源详细分析"""
        if 'interference_sources' not in self.scenario or not self.scenario['interference_sources']:
            ax.text(0.5, 0.5, '无干扰源', ha='center', va='center')
            ax.set_title('干扰源分析')
            return

        sources = self.scenario['interference_sources']

        # 绘制功率和高度散点图
        powers = [s.power for s in sources]
        heights = [s.height for s in sources]
        colors = [s.color for s in sources]  # 使用干扰源自身的颜色

        # 统一使用星型标记
        for i in range(len(sources)):
            ax.scatter(powers[i], heights[i], c=colors[i], s=100, alpha=0.7, marker='*')

        ax.set_xlabel('功率 (dBm)')
        ax.set_ylabel('高度 (m)')
        ax.set_title('干扰源功率-高度分布')
        ax.grid(True, alpha=0.3)

        # 添加图例 - 使用干扰源自身的颜色
        unique_sources = {}
        for source in sources:
            if source.name not in unique_sources:
                unique_sources[source.name] = source.color

        legend_handles = []
        for name, color in unique_sources.items():
            legend_handles.append(Line2D([0], [0], marker='*', color='w',
                                         markerfacecolor=color, markersize=10, label=name))

        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=8)

    def _plot_detailed_height_analysis(self, ax):
        """绘制详细高度分析"""
        drones = self.scenario['drone_positions']
        buildings = self.scenario['buildings']

        if len(drones) == 0:
            ax.text(0.5, 0.5, '无无人机数据', ha='center', va='center')
            ax.set_title('高度分析')
            return

        # 绘制无人机高度分布
        heights = drones[:, 2]
        x_positions = np.arange(len(heights))

        bars = ax.bar(x_positions, heights, color=self.colors['drones'], alpha=0.7)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='低空界限')
        ax.axhline(y=300, color='gray', linestyle='--', alpha=0.5, label='高空界限')

        # 标注速度
        speeds = self.scenario['drone_speeds'] if 'drone_speeds' in self.scenario else np.zeros(len(heights))
        for i, (height, speed) in enumerate(zip(heights, speeds)):
            ax.text(i, height + 5, f'{speed:.1f}m/s', ha='center', fontsize=8)

        ax.set_xlabel('无人机编号')
        ax.set_ylabel('高度 (m)')
        ax.set_title('无人机高度与速度分布')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'U{i}' for i in range(len(heights))])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_detailed_communication_analysis(self, ax):
        """绘制详细通信分析"""
        analysis = self.scenario.get('analysis', {}) if isinstance(self.scenario, dict) else {}
        model = str(analysis.get('model', '')).strip().lower()

        if analysis:
            comm_label = 'margin通信劣化（功率裕量）'
            categories = [comm_label, '速度导致的能效劣化']
            comm_deg = float(np.clip(analysis.get('avg_comm_deg', 0.0), 0.0, 1.0))
            speed_deg = float(np.clip(analysis.get('avg_speed_deg', 0.0), 0.0, 1.0))
            degradation_values = np.array([comm_deg, speed_deg], dtype=float)
        else:
            # 兜底：若没有 analysis，就保留旧的启发式画法
            categories = ['通信功率强度变化', '速度导致的通信能效(劣化)']
            sources = self.scenario.get('interference_sources', [])
            avg_power_dbm = float(np.mean([s.power for s in sources])) if sources else 0.0
            power_change = float(1.0 / (1.0 + np.exp(-(avg_power_dbm - 20.0) / 8.0)))
            speeds = self.scenario.get('drone_speeds', [])
            vmax = float(self.scenario.get('drone_speed_max', 1.0))
            vmax = max(vmax, 1e-6)
            speed_eff_deg = float(np.mean([(v / vmax) ** 2 for v in speeds])) if len(speeds) else 0.0
            speed_eff_deg = float(np.clip(speed_eff_deg / (1.0 + speed_eff_deg), 0.0, 1.0))
            degradation_values = np.array([power_change, speed_eff_deg], dtype=float)

        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        degradation_values = np.concatenate((degradation_values, [degradation_values[0]]))
        angles += angles[:1]

        ax.plot(angles, degradation_values, 'o-', linewidth=2)
        ax.fill(angles, degradation_values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 1)
        ax.set_title('通信性能劣化雷达图')
        ax.grid(True)

    def _plot_detailed_scenario_parameters(self, ax):
        """绘制详细场景参数"""
        ax.axis('off')

        drones = self.scenario['drone_positions']
        stations = self.scenario['station_positions']
        buildings = self.scenario['buildings']
        sources = self.scenario.get('interference_sources', [])

        text_lines = [
            "详细场景参数",
            "=" * 40,
            f"区域: {self.scenario['area_size']:.0f}m × {self.scenario['area_size']:.0f}m",
            f"无人机: {len(drones)}架",
            f"地面站: {len(stations)}个",
            f"建筑物: {len(buildings)}栋",
            f"干扰源: {len(sources)}个",
            "",
            "无人机参数:",
        ]

        # 无人机详细信息
        for i, drone in enumerate(drones):
            speed = self.scenario['drone_speeds'][i] if i < len(self.scenario['drone_speeds']) else 0
            text_lines.append(f"  U{i}: ({drone[0]:.1f}, {drone[1]:.1f}, {drone[2]:.1f}m) - {speed:.1f}m/s")

        text_lines.append("\n地面站位置:")
        for i, station in enumerate(stations):
            text_lines.append(f"  S{i}: ({station[0]:.1f}, {station[1]:.1f}, {station[2]:.1f}m)")

        if buildings:
            text_lines.append(f"\n建筑物统计:")
            avg_height = np.mean([b.height for b in buildings])
            max_height = np.max([b.height for b in buildings])
            text_lines.append(f"  平均高度: {avg_height:.1f}m")
            text_lines.append(f"  最大高度: {max_height:.1f}m")
            text_lines.append(f"  数量: {len(buildings)}")

        if sources:
            text_lines.append(f"\n干扰源统计:")
            ground_count = sum(1 for s in sources if s.is_ground)
            building_count = len(sources) - ground_count
            text_lines.append(f"  地面干扰源: {ground_count}个")
            text_lines.append(f"  建筑物干扰源: {building_count}个")

        text = "\n".join(text_lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    def _plot_interference_details(self, ax):
        """绘制干扰源详情"""
        sources = self.scenario.get('interference_sources', [])

        if not sources:
            ax.text(0.5, 0.5, '无干扰源', ha='center', va='center')
            ax.set_title('干扰源详情')
            return

        ax.axis('off')

        text_lines = ["干扰源详细信息", "=" * 40]

        for i, source in enumerate(sources):
            freq_ghz = source.frequency / 1e9
            location_type = "地面" if source.is_ground else "建筑物"
            building_info = f"建筑物上 ({source.building.height:.1f}m)" if source.building else "无建筑物"

            text_lines.append(f"{i + 1}. {source.name} [{location_type}]")
            text_lines.append(f"   位置: ({source.x:.1f}, {source.y:.1f}, {source.height:.1f}m)")
            text_lines.append(f"   功率: {source.power:.1f}dBm, 频率: {freq_ghz:.2f}GHz")
            if not source.is_ground:
                text_lines.append(f"   建筑物: {building_info}")
            text_lines.append("")

        text = "\n".join(text_lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))


def get_user_input():
    """获取用户输入参数"""
    print("无人机通信场景优化系统")
    print("=" * 60)
    print("请输入以下参数:")

    params = {}

    # 无人机数量
    while True:
        try:
            params['num_drones'] = int(input("无人机数量 (3-15, 推荐5-10): ") or "8")
            if 3 <= params['num_drones'] <= 15:
                break
            else:
                print("请输入3-15之间的数字")
        except ValueError:
            print("请输入有效的数字")

    # 地面站数量
    while True:
        try:
            params['num_stations'] = int(input("地面站数量 (2-8, 推荐3-5): ") or "4")
            if 2 <= params['num_stations'] <= 8:
                break
            else:
                print("请输入2-8之间的数字")
        except ValueError:
            print("请输入有效的数字")

    # 区域大小
    while True:
        try:
            params['area_size'] = float(input("区域大小 (米, 500-3000, 推荐1000-1500): ") or "1200")
            if 500 <= params['area_size'] <= 3000:
                break
            else:
                print("请输入500-3000之间的数字")
        except ValueError:
            print("请输入有效的数字")

    # 无人机最大高度
    while True:
        try:
            params['drone_height_max'] = float(input("无人机最大高度 (米, 100-500, 推荐200-400): ") or "300")
            if 100 <= params['drone_height_max'] <= 500:
                break
            else:
                print("请输入100-500之间的数字")
        except ValueError:
            print("请输入有效的数字")

    # 建筑物最大高度
    while True:
        try:
            params['building_height_max'] = float(input("建筑物最大高度 (米, 50-300, 推荐80-150): ") or "120")
            if 50 <= params['building_height_max'] <= 300:
                break
            else:
                print("请输入50-300之间的数字")
        except ValueError:
            print("请输入有效的数字")

    # 建筑覆盖率（ITU α）：用于控制建筑总覆盖面积（0~1）
    # 说明：这里的“覆盖率α”与下面的“建筑数量密度β(buildings/km^2)”不是同一个概念，避免误解为重复输入。
    while True:
        try:
            alpha = float(input("建筑覆盖率 α (0.1-0.9, 推荐0.3-0.7): ") or "0.5")
            if 0.1 <= alpha <= 0.9:
                params['building_density'] = alpha  # 保持兼容：历史字段名仍叫 building_density
                params['itu_alpha'] = alpha
                break
            else:
                print("请输入0.1-0.9之间的数字")
        except ValueError:
            print("请输入有效的数字")

    # ITU 城市统计参数（可选）：β、γ直接回车使用默认值
    # 默认 β 过大时（例如 area_size≈1200m），建筑数量会非常多、视觉上过密；
    # 这里把默认值调到更“常规演示”的水平（仍可在自定义模式里改回更高密度）。
    default_beta = 120.0
    default_gamma = max(5.0, params['building_height_max'] / 2.0)
    customize_itu = input("是否自定义ITU参数 β/γ? (y/N): ").strip().lower() == 'y'
    if customize_itu:
        # β：建筑数量密度（buildings/km^2）
        while True:
            try:
                params['itu_beta'] = float(input(
                    "ITU建筑数量密度 β (buildings/km^2, 50-2000, 回车默认120): "
                ) or "120")
                if 50 <= params['itu_beta'] <= 2000:
                    break
                else:
                    print("请输入50-2000之间的数字")
            except ValueError:
                print("请输入有效的数字")

        # γ：Rayleigh 高度分布尺度参数（m）
        while True:
            try:
                params['itu_gamma'] = float(input(
                    f"ITU建筑高度尺度 γ (m, 5-200, 回车默认{default_gamma:.1f}): "
                ) or f"{default_gamma:.1f}")
                if 5 <= params['itu_gamma'] <= 200:
                    break
                else:
                    print("请输入5-200之间的数字")
            except ValueError:
                print("请输入有效的数字")
    else:
        params['itu_beta'] = default_beta
        params['itu_gamma'] = float(default_gamma)

    # 无人机最大速度
    while True:
        try:
            params['drone_speed_max'] = float(input("无人机最大速度 (m/s, 5-30, 推荐10-20): ") or "15")
            if 5 <= params['drone_speed_max'] <= 30:
                break
            else:
                print("请输入5-30之间的数字")
        except ValueError:
            print("请输入有效的数字")

    return params


def save_scenario_data(scenario: Dict, filename: str):
    """保存场景数据到JSON文件"""
    filepath = os.path.join("output", filename)

    try:
        # 准备场景数据
        data = {
            'basic_info': {
                'area_size': float(scenario['area_size']),
                'num_drones': len(scenario['drone_positions']),
                'num_stations': len(scenario['station_positions']),
                'num_buildings': len(scenario['buildings']),
                'num_interference': len(scenario.get('interference_sources', [])),
                'building_density': float(scenario['building_density']),
                'drone_speed_max': float(scenario['drone_speed_max']),
                'drone_height_max': float(scenario['drone_height_max']),
                'building_height_max': float(scenario['building_height_max']),
                'floor_height_m': float(scenario.get('floor_height_m', 0.0))
            },
            'semantic_layers': scenario.get('semantic_layers', {}),
            'analysis': scenario.get('analysis', {}),
            'building_observables': scenario.get('building_observables', {}),
            'wifi_density': scenario.get('wifi_density', {}),
            'drones': [],
            'stations': [],
            'buildings': [],
            'interference_sources': []
        }

        # 无人机数据
        for i, drone in enumerate(scenario['drone_positions']):
            speed = scenario['drone_speeds'][i] if i < len(scenario['drone_speeds']) else 0
            data['drones'].append({
                'id': i,
                'position': [float(drone[0]), float(drone[1]), float(drone[2])],
                'speed_mps': float(speed),
                'height_m': float(drone[2])
            })

        # 地面站数据
        for i, station in enumerate(scenario['station_positions']):
            data['stations'].append({
                'id': i,
                'position': [float(station[0]), float(station[1]), float(station[2])],
                'height_m': float(station[2])
            })

        # 建筑物数据
        floor_h = max(float(scenario.get('floor_height_m', 3.0)), 0.5)
        for i, building in enumerate(scenario['buildings']):
            footprint = float(building.width) * float(building.length)
            floors = max(1, int(math.floor(float(building.height) / floor_h)))
            floor_area = float(footprint * float(floors))
            volume = float(footprint * float(building.height))
            data['buildings'].append({
                'id': i,
                'position': [float(building.x), float(building.y), float(building.height / 2)],
                'height_m': float(building.height),
                'width_m': float(building.width),
                'length_m': float(building.length),
                'color': building.color,
                'footprint_m2': float(footprint),
                'volume_m3': float(volume),
                'floors': int(floors),
                'floor_area_m2': float(floor_area)
            })

        # 干扰源数据
        for i, source in enumerate(scenario.get('interference_sources', [])):
            info = source.get_info()
            data['interference_sources'].append({
                'id': i,
                'name': info['name'],
                'type_id': info['type_id'],
                'position': info['position'],
                'power_dbm': info['power_dbm'],
                'raw_power_dbm': info.get('raw_power_dbm', info['power_dbm']),
                'profile_key': info.get('profile_key', None),
                'profile_meta': info.get('profile_meta', None),
                'power_semantics': info.get('power_semantics', None),
                'frequency_ghz': info['frequency_ghz'],
                'building_associated': info['building_associated'],
                'is_ground': info['is_ground'],
                'is_indoor': info.get('is_indoor', False),
                'indoor_floor': info.get('indoor_floor', None),
                'semantic': info.get('semantic', None),
                'activity': info.get('activity', 1.0),
                'color': info['color']
            })

        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n场景数据已保存: {filepath}")
        print("数据格式: JSON")
        print("用途: 可用于仿真模拟、进一步分析等")

    except Exception as e:
        print(f"保存场景数据时出错: {e}")


def main():
    """主函数 - 遗传算法优化"""
    print("无人机通信性能劣化场景优化 - 高级版本")
    print("=" * 60)
    print("说明:")
    print("1. 无人机高度作为染色体进行优化")
    print("2. 干扰源参数（类型、功率、位置）作为染色体进行优化")
    print("3. 干扰源可放置在建筑物或地面（地面干扰源高度为0）")
    print("4. 干扰源统一使用星型标记，不同颜色表示不同类型")
    print("=" * 60)

    # 获取用户参数
    user_params = get_user_input()

    print("\n" + "=" * 60)
    print("参数设置完成:")
    display_keys = [
        'num_drones',
        'num_stations',
        'area_size',
        'drone_height_max',
        'building_height_max',
        'building_density',
        'drone_speed_max'
    ]
    for key in display_keys:
        if key in user_params:
            print(f"  {key}: {user_params[key]}")

    # 创建问题实例
    problem = DroneCommProblem(user_params)

    # 检查维度
    if not problem.check_dimensions():
        print("维度错误，程序退出")
        return

    print("\n" + "=" * 60)
    print("开始遗传算法优化...")
    print(f"染色体维度: {problem.Dim}")
    print(f"优化目标: 最大化通信劣化程度")
    print("=" * 60)

    # 构建算法
    algorithm = ea.soea_SEGA_templet(
        problem,
        ea.Population(Encoding='RI', NIND=40),
        MAXGEN=80,
        logTras=10,
        trappedValue=1e-4,
        maxTrappedCount=10
    )

    # 开始优化
    start_time = time.time()

    try:
        res = ea.optimize(
            algorithm,
            verbose=True,
            drawing=1,
            outputMsg=True,
            drawLog=False,
            saveFlag=False
        )

        elapsed_time = time.time() - start_time
        print(f"\n优化完成，耗时: {elapsed_time:.2f}秒")

        # 获取最优解
        best_x = res['Vars'][0]
        best_fitness = res['ObjV'][0][0]

        # 打印结果
        print_optimization_results(best_x, best_fitness, problem)

        # 生成并可视化最优场景
        visualize_optimized_scenario(best_x, problem, user_params)

    except Exception as e:
        print(f"优化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def print_optimization_results(best_x: np.ndarray, best_fitness: float, problem):
    """打印优化结果"""
    print("\n" + "=" * 60)
    print("遗传算法优化结果:")
    print(f"最佳适应度（通信劣化程度）: {best_fitness:.6f}")

    print("\n染色体解码:")
    idx = 0

    # 无人机高度
    drone_heights = best_x[idx:idx + problem.num_drones]
    idx += problem.num_drones
    print(f"无人机高度 (共{len(drone_heights)}个):")
    for i, height in enumerate(drone_heights[:5]):  # 只显示前5个
        print(f"  无人机{i}: {height:.1f}m")
    if len(drone_heights) > 5:
        print(f"  ... 还有{len(drone_heights) - 5}个")

    # 无人机速度
    drone_speeds = best_x[idx:idx + problem.num_drones]
    idx += problem.num_drones
    print(f"\n无人机速度 (平均: {np.mean(drone_speeds):.1f}m/s):")

    # 无人机水平位置
    drone_xy = best_x[idx:idx + problem.num_drones * 2].reshape(problem.num_drones, 2)
    idx += problem.num_drones * 2
    print(f"\n无人机水平位置 (前{min(5, problem.num_drones)}个):")
    for i, (x_pos, y_pos) in enumerate(drone_xy[:5]):
        print(f"  无人机{i}: ({x_pos:.1f}, {y_pos:.1f})")
    if problem.num_drones > 5:
        print(f"  ... 还有{problem.num_drones - 5}个")

    # 干扰源类型
    interference_types = [int(t) for t in best_x[idx:idx + problem.num_interference]]
    idx += problem.num_interference

    # 干扰源功率
    interference_powers = best_x[idx:idx + problem.num_interference]
    idx += problem.num_interference

    # 干扰源位置类型
    interference_location_types = [int(t) for t in best_x[idx:idx + problem.num_interference]]
    idx += problem.num_interference

    config = INTERFERENCE_SOURCE_CONFIG
    print(f"\n干扰源配置 (共{len(interference_types)}个):")
    for i, (type_id, power, location_type) in enumerate(
            zip(interference_types, interference_powers, interference_location_types)):
        type_config = config.get_config(type_id)
        location_str = "地面" if location_type == 0 else "建筑物"
        print(f"  干扰源{i}: {type_config['name']}, 功率: {power:.1f}dBm, 位置: {location_str}")


def visualize_optimized_scenario(best_x: np.ndarray, problem, user_params: Dict):
    """可视化最优场景"""
    print("\n" + "=" * 60)
    print("生成场景可视化...")

    # 生成场景
    scenario = problem.generate_scenario(best_x)
    # 为可视化附加通信分析（干扰强度/链路裕量等；已统一使用 margin 劣化模型）
    try:
        scenario['analysis'] = problem.analyze_scenario_metrics(scenario)
    except Exception as e:
        print(f"场景分析失败（不影响可视化）：{e}")

    print(f"\n场景统计:")
    print(f"  无人机数量: {len(scenario['drone_positions'])}")
    print(f"  地面站数量: {len(scenario['station_positions'])}")
    print(f"  建筑物数量: {len(scenario['buildings'])}")
    print(f"  干扰源数量: {len(scenario['interference_sources'])}")
    print(f"  区域大小: {scenario['area_size']:.0f}m × {scenario['area_size']:.0f}m")

    # 无人机统计
    drone_heights = scenario['drone_positions'][:, 2]
    drone_speeds = scenario['drone_speeds']
    print(
        f"  无人机平均高度: {np.mean(drone_heights):.1f}m (范围: {np.min(drone_heights):.1f}-{np.max(drone_heights):.1f}m)")
    print(f"  无人机平均速度: {np.mean(drone_speeds):.1f}m/s")

    # 建筑物统计
    if scenario['buildings']:
        building_heights = [b.height for b in scenario['buildings']]
        print(f"  建筑物平均高度: {np.mean(building_heights):.1f}m")
        print(f"  建筑物最大高度: {np.max(building_heights):.1f}m")

    # 干扰源统计
    if scenario['interference_sources']:
        interference_types = {}
        ground_count = 0
        building_count = 0

        for source in scenario['interference_sources']:
            name = source.name
            interference_types[name] = interference_types.get(name, 0) + 1
            if source.is_ground:
                ground_count += 1
            else:
                building_count += 1

        print(f"\n干扰源类型分布:")
        for name, count in interference_types.items():
            print(f"  {name}: {count}个")

        print(f"\n干扰源位置分布:")
        print(f"  地面干扰源: {ground_count}个")
        print(f"  建筑物干扰源: {building_count}个")

    # 创建可视化器
    visualizer = EnhancedVisualizer(scenario)

    # 先生成单独的图表
    print("\n生成单独的图表...")
    plots_info = visualizer.create_individual_plots("optimized")

    print("\n已生成以下图表:")
    for name, filename in plots_info:
        print(f"  {name}: {filename}")

    # 保存场景数据
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    data_filename = f"scenario_data_{timestamp}.json"
    save_scenario_data(scenario, data_filename)

    # 询问是否生成综合报告
    print("\n是否生成综合报告图? (y/n)")
    choice = input("请输入选择: ").lower()

    if choice == 'y':
        print("生成综合报告...")
        report_file = visualizer.create_comprehensive_report("optimized")
        print(f"综合报告已保存: {report_file}")

    # 分析通信劣化
    print("\n" + "=" * 60)
    print("通信劣化分析:")

    # 计算平均劣化程度
    total_degradation = 0
    link_count = 0

    for drone_idx in range(len(scenario['drone_positions'])):
        for station_idx in range(len(scenario['station_positions'])):
            degradation = problem.calculate_link_degradation(
                scenario['drone_positions'][drone_idx],
                scenario['station_positions'][station_idx],
                scenario,
                drone_speed_mps=float(scenario['drone_speeds'][drone_idx]),
                drone_idx=int(drone_idx)
            )
            total_degradation += degradation
            link_count += 1

    if link_count > 0:
        avg_degradation = total_degradation / link_count
        print(f"  平均通信劣化程度: {avg_degradation:.3f}")

        # 评级
        if avg_degradation > 0.7:
            rating = "严重劣化"
            suggestion = "强烈建议重新规划飞行路径和通信频段"
        elif avg_degradation > 0.5:
            rating = "显著劣化"
            suggestion = "建议优化无人机高度和避让干扰源"
        elif avg_degradation > 0.3:
            rating = "中等劣化"
            suggestion = "通信质量可接受，建议监控关键链路"
        else:
            rating = "轻微劣化"
            suggestion = "通信质量良好"

        print(f"  劣化评级: {rating}")
        print(f"  建议: {suggestion}")

    print("\n" + "=" * 60)
    print("优化完成！所有结果已保存到output目录")
    print("=" * 60)


def quick_demo():
    """快速演示"""
    print("无人机通信场景快速演示")
    print("=" * 60)
    print("使用预设参数生成示例场景")
    print("=" * 60)

    # 预设参数
    user_params = {
        'num_drones': 6,
        'num_stations': 4,
        'area_size': 1200,
        'drone_height_max': 350,
        # 让预设场景“更正常一些”：控制建筑数量密度 β 与覆盖率 α，避免过密
        'building_height_max': 120,
        'building_density': 0.45,  # 历史字段名；等价于 ITU α
        'itu_alpha': 0.45,
        'itu_beta': 120.0,
        'itu_gamma': 40.0,
        'drone_speed_max': 18
    }

    print("预设参数:")
    for key, value in user_params.items():
        print(f"  {key}: {value}")

    # 创建问题实例
    problem = DroneCommProblem(user_params)

    # 使用预设染色体（模拟优化结果）
    np.random.seed(123)
    preset_x = np.random.uniform(problem.lb, problem.ub)

    # 整数参数取整
    type_start = problem.num_drones * 4
    preset_x[type_start:type_start + problem.num_interference] = \
        np.round(preset_x[type_start:type_start + problem.num_interference])
    loc_start = problem.num_drones * 4 + problem.num_interference * 2
    preset_x[loc_start:loc_start + problem.num_interference] = \
        np.round(preset_x[loc_start:loc_start + problem.num_interference])

    # 生成并可视化场景
    visualize_optimized_scenario(preset_x, problem, user_params)


if __name__ == "__main__":
    # 选择运行模式
    print("无人机通信场景优化与可视化系统")
    print("=" * 60)
    print("请选择运行模式:")
    print("1. 完整遗传算法优化")
    print("2. 快速演示（使用预设参数）")
    print("3. 退出")

    if sys.stdin is None or not sys.stdin.isatty():
        print("\n(未检测到可交互输入，默认运行：快速演示)")
        choice = "2"
    else:
        try:
            choice = input("请输入选择 (1-3): ")
        except (EOFError, KeyboardInterrupt):
            # Non-interactive stdin (some IDE runners / piping) can't read input().
            # Fall back to quick demo so the script can still run.
            print("\n(未检测到可交互输入，默认运行：快速演示)")
            choice = "2"

    if choice == "1":
        main()
    elif choice == "2":
        quick_demo()
    elif choice == "3":
        print("程序退出")
    else:
        print("无效选择，运行完整遗传算法优化...")
        main()
