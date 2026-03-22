from __future__ import annotations

from pathlib import Path
from textwrap import dedent


UAV_GA = r"D:\论文无人机\成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\UAV_GA.py"
UAV_GAN = r"D:\论文无人机\成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\gan_uav_pipeline.py"
UAV_CMP = r"D:\论文无人机\成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\compare_random_ga_gan.py"
UAV_EVAL = r"D:\论文无人机\成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\evaluate.py"
UAV_KPI = r"D:\论文无人机\成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\kpi.py"
UAV_MEAS = r"D:\论文无人机\成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\compare_measured_cornercases.py"
UAV_BLER_MC = r"D:\论文无人机\成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\bler_mc.py"
UAV_BLER_SIONNA = r"D:\论文无人机\成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\bler_sionna.py"

SAT_MAIN = r"D:\论文卫星\成果本身\代码工程\LEO_Sim\v7proj\LEO_StarNet_EMC_V7_0_Engineering.m"
SAT_CFG = r"D:\论文卫星\成果本身\代码工程\LEO_Sim\v7proj\emcDefaultConfig.m"
SAT_BUILD = r"D:\论文卫星\成果本身\代码工程\LEO_Sim\v7proj\emcBuildLinkModel.m"
SAT_SIM = r"D:\论文卫星\成果本身\代码工程\LEO_Sim\v7proj\simulateStarNetV7.m"
SAT_COMP = r"D:\论文卫星\成果本身\代码工程\LEO_Sim\v7proj\emcComputeComplianceRowsV7.m"
SAT_SUM = r"D:\论文卫星\成果本身\代码工程\LEO_Sim\v7proj\emcWriteSummaryTextV7.m"
SAT_GAN = r"D:\论文卫星\成果本身\代码工程\LEO_Sim\v7proj\trainOrLoadJammerGAN.m"
SAT_WORST = r"D:\论文卫星\成果本身\代码工程\LEO_Sim\v7proj\worstCaseObjectiveV7.m"
SAT_CLS = r"D:\论文卫星\成果本身\代码工程\LEO_Sim\v7proj\classifyInterferenceTimeline_powerSampler.m"
SAT_DATA = r"D:\论文卫星\成果本身\代码工程\LEO_Sim\v7proj\generateDatasetSimpleSTFT.m"
SAT_TRAIN = r"D:\论文卫星\成果本身\代码工程\LEO_Sim\v7proj\trainLeNetSTFT.m"

PROTO_BUILD = r"D:\论文卫星\成果本身\代码工程\LEO_EMC_Sim_原型与Simulink\build_LEO_EMC_Sim.m"
PROTO_CFO = r"D:\论文卫星\成果本身\代码工程\LEO_EMC_Sim_原型与Simulink\LEO_EMC_Sim\cfo_comp.m"
PROTO_INTF = r"D:\论文卫星\成果本身\代码工程\LEO_EMC_Sim_原型与Simulink\LEO_EMC_Sim\interf_gen.m"
PROTO_V1 = r"D:\论文卫星\成果本身\代码工程\LEO_EMC_Sim_原型与Simulink\LEO_EMC_Sim\LEO_Sim_V1.m"
PROTO_RUN = r"D:\论文卫星\成果本身\代码工程\LEO_EMC_Sim_原型与Simulink\LEO_EMC_Sim\run_LEO_EMC_Sim.m"
PROTO_PRE = r"D:\论文卫星\成果本身\代码工程\LEO_EMC_Sim_原型与Simulink\LEO_EMC_Sim\preamble_insert.m"

OWN_BUILD = r"D:\专利自有\成果本身\代码工程\一汽项目_卫星EMC原型工程与数据集\build_LEO_EMC_Sim.m"
OWN_CFO = r"D:\专利自有\成果本身\代码工程\一汽项目_卫星EMC原型工程与数据集\LEO_EMC_Sim\cfo_comp.m"
OWN_INTF = r"D:\专利自有\成果本身\代码工程\一汽项目_卫星EMC原型工程与数据集\LEO_EMC_Sim\interf_gen.m"
OWN_V1 = r"D:\专利自有\成果本身\代码工程\一汽项目_卫星EMC原型工程与数据集\LEO_EMC_Sim\LEO_Sim_V1.m"
OWN_RUN = r"D:\专利自有\成果本身\代码工程\一汽项目_卫星EMC原型工程与数据集\LEO_EMC_Sim\run_LEO_EMC_Sim.m"
OWN_PRE = r"D:\专利自有\成果本身\代码工程\一汽项目_卫星EMC原型工程与数据集\LEO_EMC_Sim\preamble_insert.m"

HEADING = "## 原文-工程行级对应与分析"


def section(text: str) -> str:
    return dedent(text).strip() + "\n"


SECTIONS: dict[Path, str] = {
    Path(r"D:\论文无人机\obsidian知识库\10_成果节点\无人机研究_代码工程与设计文档.md"): section(
        f"""
        {HEADING}

        ### 1. 干扰源类型化标定
        - 原文抓手：`数据标定.docx`、`干扰源数据标定.docx` 中对 Wi-Fi / 4G / 5G 干扰源的频段、EIRP、活跃度和来源说明。
        - 工程对应：`{UAV_GA}:54-120`；`{UAV_GA}:2241-2303`
        - 分析：`InterferenceSourceConfig` 先把不同干扰源的功率范围、频段、覆盖半径和 calibration 字段固化下来，随后 `_precompute_interference_power_mw_per_drone()` 把这些标定值真正折算成每架无人机的总干扰功率，所以这批文档是 `I_y_mw` 的直接参数来源。

        ### 2. 城市场景与语义层
        - 原文抓手：`城市模型生成.docx`、`代码.docx` 里关于建筑密度、楼层统计、道路/人行道/屋顶/室内语义分层的描述。
        - 工程对应：`{UAV_GA}:625-680`；`{UAV_GA}:1101-1180`；`{UAV_GA}:1484-1565`
        - 分析：`DroneCommProblem.__init__()` 固定了 ITU 城市统计参数，`_compute_building_observables()` 把建筑物转成楼层、面积和体积统计，`_generate_semantic_layers()` 再把道路、屋顶、立面、杆塔等语义约束落到可计算的几何层，因此文档中的“城市模型生成”与这里是同一条实现链。

        ### 3. 功率域链路与路径损耗
        - 原文抓手：论文级报告中“A2G / U2U / 室内穿透 / LoS-NLoS / SINR / 能效”的公式段。
        - 工程对应：`{UAV_GA}:2412-2472`；`{UAV_GA}:2793-2865`；`{UAV_GA}:3065-3115`；`{UAV_GA}:3500-3555`；`{UAV_GA}:3572-3625`
        - 分析：`_interference_path_loss_db()` 实现了 LoS/NLoS 加权和室内 P.1238 / P.2109 的损耗叠加，`calculate_power_margin_components()` 给出 `P_rx / I_y / N_y / P_IN / M / SINR / R` 的统一功率域分解，推进功率与空中干扰则分别由 `_rotor_propulsion_power_w()` 和 `calculate_aerial_interference_power_mw()` 补齐，因此报告中的主要公式都能在这些行找到直接实现。

        ### 4. 最坏场景搜索与结果输出
        - 原文抓手：`0315` 工程和“最终代码/代码详解”里关于 GA、GAN、KPI、BLER 和比较实验的说明。
        - 工程对应：`{UAV_GAN}:78-199`；`{UAV_CMP}:75-245`；`{UAV_EVAL}:102-150`；`{UAV_EVAL}:243-328`；`{UAV_KPI}:58-123`
        - 分析：`run_ga_samples()` 负责搜最坏样本，`train_gan()` 与 `generate_gan_samples()` 扩展极端样本分布，`evaluate_groups_from_samples()` 统一生成 `BLER_A / BLER_B / KPI / throughput`，`compute_kpis()` 与 `compute_throughput_kpis()` 再把链路记录转成论文可直接引用的统计指标，这正是新增代码资产和已有论文无人机主工程的衔接点。
        """
    ),
    Path(r"D:\论文无人机\obsidian知识库\10_成果节点\无人机研究_论文报告与汇报材料.md"): section(
        f"""
        {HEADING}

        ### 1. “联合优化飞行状态与受控干扰配置” -> 优化问题主定义
        - 原文句子：`联合优化无人机飞行状态与受控干扰配置，使通信链路指标出现可度量的劣化。`
        - 工程对应：`{UAV_GA}:625-680`；`{UAV_GAN}:78-199`；`{UAV_CMP}:75-245`
        - 分析：这里的表述不是抽象提法，而是直接对应 `DroneCommProblem` 的变量边界与场景参数，以及 GA 采样、GAN 学习、GA/GAN/Random 三组比较的整条实验链。

        ### 2. “路径损耗、遮挡判定、LoS/NLoS” -> 功率域链路实现
        - 原文句子：`模型需要同时支持 A2G、U2U 与 UAV-干扰源链路，并考虑建筑遮挡与室内/穿透损耗。`
        - 工程对应：`{UAV_GA}:2412-2472`；`{UAV_GA}:2305-2325`；`{UAV_GA}:2801-2865`；`{UAV_GA}:3572-3625`
        - 分析：`_interference_path_loss_db()` 与 `calculate_los_probability()` 处理遮挡和 LoS 口径，`calculate_power_margin_components()` 和 `compute_sinr_zeng()` 则把这些口径统一压成可比较的 `SINR / rate / outage` 输出，所以报告中的公式段已经被实现为可执行的链路模型。

        ### 3. “BLER/KPI/吞吐” -> 结果层生成代码
        - 原文句子：`后续结果需落到 BLER、吞吐量、可靠性和能效指标。`
        - 工程对应：`{UAV_BLER_MC}:101-140`；`{UAV_BLER_SIONNA}:33-114`；`{UAV_EVAL}:243-328`；`{UAV_KPI}:58-123`
        - 分析：`simulate_bler_curve()` 与 `simulate_bler_curve_sionna()` 形成两套 BLER LUT，`evaluate_groups_from_samples()` 把场景评价落到链路 CSV 与 `kpi_report.json`，`compute_kpis()` / `compute_throughput_kpis()` 把指标汇总成论文图表与定量结论。

        ### 4. “实测对照与极限场景” -> 实测坏点复现实验
        - 原文句子：`测量对照验证`、`极限场景生成` 和组会里反复出现的“坏点对照”“角点对照”。
        - 工程对应：`{UAV_MEAS}:530-640`；`{UAV_MEAS}:1186-1305`
        - 分析：`_build_problem()`、`_attach_aligned_environment()` 和 `run_comparison()` 负责把 AERPAW / Dryad 数据对齐到本地问题定义、附加经验热点、生成 GA/GAN/随机三类候选，并输出 `grid_map / worst_region / corner_case_report`，这就是汇报材料里的“实测-建模-对照”闭环。
        """
    ),
    Path(r"D:\论文无人机\obsidian知识库\20_参考节点\无人机研究_专题论文与参考资料.md"): section(
        f"""
        {HEADING}

        ### 1. A2G / LoS / 城市随机几何文献 -> 城市与链路模型
        - 原文抓手：`1220论文`、`论文` 目录里的 Al-Hourani、A2G path loss、Optimal LAP、3D stochastic 等论文。
        - 工程对应：`{UAV_GA}:625-680`；`{UAV_GA}:2305-2325`；`{UAV_GA}:2412-2495`
        - 分析：这些文献在代码里具体落成了 ITU 城市统计参数、Sigmoid/乘积两套 LoS 口径以及 LoS/NLoS 路损加权，因此不是泛泛“参考文献”，而是 `calculate_los_probability()` 和 `_interference_path_loss_db()` 的依据层。

        ### 2. 干扰管理与频谱耦合文献 -> 干扰源库与聚合模型
        - 原文抓手：`5G_and_Beyond_Interference_Management`、Wi-Fi/蜂窝干扰与 EMC 相关文献。
        - 工程对应：`{UAV_GA}:54-120`；`{UAV_GA}:2241-2303`；`{UAV_GA}:3500-3555`
        - 分析：文献中关于干扰类型、频谱重叠和耦合强度的讨论，在代码里分别体现在 `InterferenceSourceConfig`、地面干扰聚合和空中干扰聚合三处，最终共同决定 `I_total_mw` 的计算口径。

        ### 3. 能耗与轨迹优化文献 -> 能效指标实现
        - 原文抓手：`Energy-Efficient_UAV_Communication...`、旋翼无人机能耗与轨迹优化论文。
        - 工程对应：`{UAV_GA}:3065-3115`；`{UAV_EVAL}:102-150`；`{UAV_KPI}:58-123`
        - 分析：推进功率模型 `_rotor_propulsion_power_w()` 是把文献中的旋翼功率分解落地到 bit/J 口径的关键，后续 `extract_link_records()` 和 `compute_kpis()` 才能把速率与推进功率合并成能效指标。

        ### 4. 实测研究与公开数据文献 -> 角点对照验证
        - 原文抓手：AERPAW、Dryad、测量类论文和 `1222论文` 里的测试标准资料。
        - 工程对应：`{UAV_MEAS}:530-640`；`{UAV_MEAS}:1186-1305`
        - 分析：新增文献资产推动了 `compare_measured_cornercases.py` 这条线，把公开测量点、热点对齐和模型坏点搜索整合进同一个对比流程，使“论文参考”能够反馈到工程验证而不是停留在综述层。
        """
    ),
    Path(r"D:\论文无人机\obsidian知识库\20_参考节点\无人机研究_测试标定与实验建设资料.md"): section(
        f"""
        {HEADING}

        ### 1. 干扰源数据标定 -> 干扰源配置表
        - 原文抓手：`干扰源数据标定.docx`、`密度依据.docx`、证据截图包中的频段/功率/密度口径。
        - 工程对应：`{UAV_GA}:54-120`
        - 分析：这部分资料最直接地进入 `InterferenceSourceConfig` 的 `frequency / power_range / coverage_radius / calibration` 字段，决定后续场景生成时不同干扰源的物理边界。

        ### 2. 建设方案与场景一致性 -> 楼层统计和语义投影
        - 原文抓手：`实验室建设方案.docx`、`无人机通信测试梳理.docx` 中对场景组织、道路/建筑/立面/室内位置的约束。
        - 工程对应：`{UAV_GA}:1101-1180`；`{UAV_GA}:1484-1565`
        - 分析：工程里用楼层面积统计和语义层投影把“实验环境如何摆放、热点如何落位”的口径转成可运算几何约束，因此这些建设材料与场景生成模块是直接耦合的。

        ### 3. 标定后的链路评估 -> 功率分解与 KPI
        - 原文抓手：资料里对“标定后如何评估干扰、可靠性和吞吐”的说明。
        - 工程对应：`{UAV_GA}:2801-2865`；`{UAV_EVAL}:243-328`；`{UAV_KPI}:58-123`
        - 分析：`calculate_power_margin_components()` 先输出 `P_rx / I_y / N_y / M / SINR / R`，再由 `evaluate_groups_from_samples()` 和 KPI 模块把标定过的场景批量转成论文可用的统计结果，这就是“测试评估”在工程上的落点。

        ### 4. 实测坏点与实验对齐 -> measured corner cases
        - 原文抓手：证据包、截图包以及实验室建设材料里的“热点/坏点/测试路线”。
        - 工程对应：`{UAV_MEAS}:530-640`；`{UAV_MEAS}:1186-1305`
        - 分析：`compare_measured_cornercases.py` 使用环境对齐、热点附着和坏点搜索去复现实测弱点，因此新增的实验与标定资产主要服务这条验证链，而不是独立于主工程存在。
        """
    ),
    Path(r"D:\论文卫星\obsidian知识库\10_成果节点\卫星EMC原型工程与Simulink模型.md"): section(
        f"""
        {HEADING}

        ### 1. “自动搭建 Simulink 通信链” -> 模型生成器
        - 原文抓手：原型工程说明报告中“自动建模、R2021a 兼容、通信视角 Simulink”的表述。
        - 工程对应：`{PROTO_BUILD}:1-20`；`{PROTO_BUILD}:74-205`
        - 分析：`build_LEO_EMC_Sim()` 负责建模入口与时间步配置，`buildTX / buildChannelEMC / buildRX / buildMetrics` 把发射、信道干扰、接收和指标输出按块自动装配，所以该原型节点与报告文字是逐段对应的。

        ### 2. “干扰注入 + CFO 补偿” -> 关键接收机算法
        - 原文抓手：说明文档和根目录简版材料里对同频噪声、单音、脉冲、邻频、同址耦合干扰，以及 CFO 估计/补偿的描述。
        - 工程对应：`{PROTO_BUILD}:305-360`；`{PROTO_INTF}:1-37`；`{PROTO_CFO}:1-23`
        - 分析：`interf_gen` 负责五类等效干扰波形，`cfo_comp` 用前导相关估计并补偿频偏，这两条就是原型工程里“EMC 干扰 + 接收机恢复”的最小可运行闭环。

        ### 3. “物理层到链路评估” -> V1 原始仿真脚本
        - 原文抓手：早期说明报告中对轨道、信道、干扰、蒙特卡罗误码和吞吐曲线的描述。
        - 工程对应：`{PROTO_V1}:123-178`；`{PROTO_V1}:229-321`；`{PROTO_V1}:585-634`
        - 分析：`defaultConfig()` 固定场景、链路和干扰参数，`computeCNI()` 先算 `C/N/I/SINR`，`runMonteCarlo()` 与 `genInterferenceWaveform()` 再把这些功率量真正压成 BER/BLER/THR 结果，因此新增原型资产是 V7 工程化之前的底层证据链。

        ### 4. “参数可调的一键运行” -> 原型执行入口
        - 原文抓手：说明报告中“修改 cfg 即可复现实验”的说法。
        - 工程对应：`{PROTO_RUN}:1-15`；`{PROTO_PRE}:1-25`
        - 分析：`run_LEO_EMC_Sim()` 只暴露 `cfg.emc.type / JS_dB / fD_Hz / cfoMethod` 等关键参数，`preamble_insert()` 负责帧结构组织，这说明原型工程的目标不是做复杂 UI，而是作为可快速复现实验的最小工程载体。
        """
    ),
    Path(r"D:\论文卫星\obsidian知识库\10_成果节点\卫星干扰识别数据与训练模型.md"): section(
        f"""
        {HEADING}

        ### 1. 数据集构造 -> STFT 数据生成脚本
        - 原文抓手：新增数据资产 `dataset_stft_r2021a` 与导出图 `montage_train.png / confusion_test.png / sim_keyframes_montage.png`。
        - 工程对应：`{SAT_DATA}:1-20`
        - 分析：`generateDatasetSimpleSTFT()` 明确给出 `none / tone / pbnj / mod` 四类标签、`train/val/test` 划分、样本数、SNR 和 JSR 采样范围，所以数据集的目录结构和数量分布都是由这里生成的。

        ### 2. 分类网络训练 -> LeNet 训练入口
        - 原文抓手：新增训练模型 `lenet_stft_model_r2021a.mat`。
        - 工程对应：`{SAT_TRAIN}:1-45`
        - 分析：`imageDatastore -> augmentedImageDatastore -> LeNet layers -> trainNetwork` 构成完整训练链，导入的 `mat` 模型是这一训练脚本输出的参数快照，而不是孤立文件。

        ### 3. 主仿真时间轴分类 -> 在线识别链
        - 原文抓手：导出图里出现的关键帧和混淆矩阵。
        - 工程对应：`{SAT_CLS}:1-84`；`{SAT_MAIN}:392-405`
        - 分析：主工程在最坏场景仿真后调用 `classifyInterferenceTimeline_powerSampler()`，用 `PS / PI / PJ / PN` 重建 IQ、做 STFT、再交给 LeNet 分类，同时按关键帧导出图片，所以 `_exports` 目录下的图片与这些代码是一一对应的。

        ### 4. 资产在工程中的角色
        - 原文抓手：该节点既包含 `dataset_stft_r2021a`，又包含 `GAN_Jammer_R2021a.mat` 与 `lenet_stft_model_r2021a.mat`。
        - 工程对应：`{SAT_CFG}:115-118`；`{SAT_GAN}:1-90`；`{SAT_WORST}:1-57`
        - 分析：配置文件把分类数据根目录固定为 `dataset_stft_r2021a`，而最坏工况搜索又单独依赖 InfoGAN 干扰生成器，这说明导入的数据资产实际上填补的是“识别支链”和“干扰生成支链”两条不同工程路线。
        """
    ),
    Path(r"D:\论文卫星\obsidian知识库\20_参考节点\一汽项目_技术报告与简版方案.md"): section(
        f"""
        {HEADING}

        ### 1. “实测-建模-仿真-智能生成-验证闭环” -> 主工程主链
        - 原文句子：一页纸/两页纸里反复出现的 `实测-建模-仿真-智能生成-验证闭环`。
        - 工程对应：`{SAT_MAIN}:281-341`；`{SAT_MAIN}:392-405`
        - 分析：基线与最坏工况仿真、InfoGAN 干扰生成、GA 搜索和 STFT 分类都集中在这两段主流程里，所以简版材料里的方法总图可以直接拆解到这两个代码区块。

        ### 2. “标准阈值与联合评估” -> 配置与判定表
        - 原文句子：`通信性能与 EMC（骚扰/抗扰度）联合评估`。
        - 工程对应：`{SAT_CFG}:120-163`；`{SAT_COMP}:1-53`
        - 分析：阈值、测量换算占位、输出目录等都由 `emcDefaultConfig()` 统一配置，`emcComputeComplianceRowsV7()` 再把 `SINR / 吞吐 / 灵敏度 / 多普勒 / Ku 指标 / JA3700` 映射成 PASS/FAIL 行表，这就是“联合评估”的代码化形式。

        ### 3. “输出评估报告与摘要” -> 文本交付层
        - 原文句子：简版材料中的 `输出用例库、评分与防护建议`、`评估报告模板`。
        - 工程对应：`{SAT_SUM}:10-38`；`{SAT_MAIN}:853-854`
        - 分析：工程最终把关键配置、基线/最坏吞吐和阈值写到 `summary_v7.txt`，因此文档里的报告型表述不是额外加工，而是由摘要输出层直接支持。

        ### 4. “原型验证” -> Simulink 原型工程
        - 原文句子：技术报告中对实验室仿真原型、节点库、接口函数的描述。
        - 工程对应：`{PROTO_BUILD}:47-68`；`{PROTO_BUILD}:107-182`
        - 分析：原型工程负责最小可运行的干扰注入、接收补偿和指标输出，是简版方案里“先有原型、再上工程化 V7 平台”的底座证据。
        """
    ),
    Path(r"D:\论文卫星\obsidian知识库\20_参考节点\一汽项目_参考文献与标准合集.md"): section(
        f"""
        {HEADING}

        ### 1. 标准阈值 -> 配置文件
        - 原文抓手：`参考文献` 目录中的车载卫星通信 EMC 标准、研究方案定稿和相关专利资料。
        - 工程对应：`{SAT_CFG}:120-163`
        - 分析：标准里的 SINR、吞吐、灵敏度、多普勒率、Ku 指标和 JA3700 等级，在工程里统一收敛到 `cfg.Requirements`，所以这批参考资产的核心作用是给配置层提供门限口径。

        ### 2. 标准条款 -> 合规判定行表
        - 原文抓手：对标与标准比较材料中的“满足 / 不满足”“对照关系”。
        - 工程对应：`{SAT_COMP}:8-53`
        - 分析：`emcComputeComplianceRowsV7()` 把每一条技术约束展开成可读的 PASS/FAIL 行表，是标准文本进入工程判定的直接落点。

        ### 3. 工程输出 -> 摘要与路由仿真
        - 原文抓手：资料中对通信质量、路测口径和结果摘要的要求。
        - 工程对应：`{SAT_BUILD}:1-35`；`{SAT_SIM}:1-90`；`{SAT_SUM}:10-38`
        - 分析：链路参数先由 `emcBuildLinkModel()` 封装，再由 `simulateStarNetV7()` 求出时序结果，最后 `emcWriteSummaryTextV7()` 输出摘要文本，这使参考文献里的“条款/口径”与工程里的“结果/摘要”形成闭环。
        """
    ),
    Path(r"D:\正向设计规范\obsidian知识库\10_成果节点\一汽项目_根目录方案简版与定稿.md"): section(
        f"""
        {HEADING}

        ### 1. 正向设计的门限骨架
        - 原文抓手：根目录定稿与一页纸/两页纸方案中的 `设计初期系统性考虑 EMC`、`满足标准门限`。
        - 工程对应：`{SAT_CFG}:120-163`
        - 分析：这些方案文档中的门限与设计目标，在工程里集中体现在 `cfg.Requirements` 与 `cfg.Measurement`，说明规范语言已经被压成可执行配置，而不是停留在概念层。

        ### 2. “系统架构 + 最坏工况” -> 搜索与仿真主链
        - 原文抓手：方案里对系统级协同、最不利工况和多源干扰搜索的描述。
        - 工程对应：`{SAT_MAIN}:281-341`；`{SAT_WORST}:1-57`
        - 分析：V7 主工程先做基线仿真，再调用 InfoGAN + GA 搜最坏样本；`worstCaseObjectiveV7()` 明确了吞吐、失效率、BLER 与能量惩罚的目标函数，所以方案中的“系统级最坏工况”在这里是有数学目标的。

        ### 3. “条款 -> 判定表 -> 摘要输出” -> 规范化交付链
        - 原文抓手：方案中的规范条文、评估结论和交付要求。
        - 工程对应：`{SAT_COMP}:1-53`；`{SAT_SUM}:10-38`；`{SAT_MAIN}:853-854`
        - 分析：判定表和摘要文本是正向设计规范最直接的工程化出口，因此该批方案文档与其说是孤立材料，不如说是 `cfg + compliance rows + summary` 三层输出的文字镜像。

        ### 4. 原型级验证支撑
        - 原文抓手：根目录材料里关于“实验室仿真技术”“节点库”“接口函数”的表述。
        - 工程对应：`{PROTO_BUILD}:47-68`；`{PROTO_BUILD}:107-182`
        - 分析：原型工程提供了最小可复现实验链，是定稿方案能够落到工程实现而不是纯理论方案的前置支撑。
        """
    ),
    Path(r"D:\正向设计规范\obsidian知识库\20_参考节点\一汽项目_项目总结与学位论文材料.md"): section(
        f"""
        {HEADING}

        ### 1. 项目总结里的“平台演进” -> V1 到 V7
        - 原文抓手：项目总结与学位论文中关于仿真平台、链路建模和试验验证能力演进的叙述。
        - 工程对应：`{PROTO_V1}:123-178`；`{PROTO_V1}:229-321`；`{SAT_MAIN}:1-40`
        - 分析：V1 脚本给出早期场景、C/N/I 和蒙特卡罗链路，V7 工程版再把这些能力扩展成配置化平台和交付版入口，因此总结材料中的“阶段演进”与两代工程代码能直接对上。

        ### 2. 项目层量化评价 -> 仿真与判定
        - 原文抓手：总结材料里的性能量化、指标评价和结论段。
        - 工程对应：`{SAT_SIM}:1-90`；`{SAT_COMP}:8-53`
        - 分析：真正的量化链路由 `simulateStarNetV7()` 产生时序 `SINR / THR / BLER`，再由 `emcComputeComplianceRowsV7()` 压成判定表，所以总结类材料里的“结论”可以回溯到具体求解与判定代码。

        ### 3. 智能算法与识别链 -> GAN + LeNet
        - 原文抓手：项目材料中对智能生成和干扰识别的讨论。
        - 工程对应：`{SAT_GAN}:1-90`；`{SAT_CLS}:1-84`
        - 分析：InfoGAN 负责最坏干扰包络生成，LeNet 负责对仿真时间序列的 STFT 图像分类，两者共同构成项目总结里“智能算法支撑 EMC 仿真”的真实工程落点。

        ### 4. 最终摘要交付
        - 原文抓手：项目总结里的总结页和学位论文中的结果汇总语句。
        - 工程对应：`{SAT_SUM}:10-38`
        - 分析：摘要输出函数把关键频点、基线/最坏性能和门限信息写成固定格式文本，是从长篇项目材料提炼出对外交付结论的最后一步。
        """
    ),
    Path(r"D:\正向设计规范\obsidian知识库\20_参考节点\一汽项目_标准与对标参考合集.md"): section(
        f"""
        {HEADING}

        ### 1. 标准条款的工程化入口
        - 原文抓手：合集中的标准、专利对比、车载卫星通信相关资料。
        - 工程对应：`{SAT_CFG}:120-163`
        - 分析：标准和对标文件里最核心的其实是阈值、测量占位和输出口径，这些都被 `emcDefaultConfig()` 收口到统一配置层，保证后续所有仿真共用同一门限体系。

        ### 2. 对标结论如何变成“是否通过”
        - 原文抓手：材料中的“满足标准 / 不满足标准 / 与专利口径一致”的比较语句。
        - 工程对应：`{SAT_COMP}:8-53`
        - 分析：`emcComputeComplianceRowsV7()` 就是把这些语句形式化成机器可判定的行表，所以该合集对规范线的真正价值在于把文字对标变成可执行对标。

        ### 3. 对外输出与口径统一
        - 原文抓手：合集中的研究方案定稿和汇报型资料。
        - 工程对应：`{SAT_SUM}:30-38`；`{SAT_MAIN}:853-854`
        - 分析：要求、现状与最终结论最终都会汇入摘要输出和导出文件，所以这批参考资料本质上约束的是“结果怎样写”和“阈值怎样解释”。
        """
    ),
    Path(r"D:\正向设计规范\obsidian知识库\20_参考节点\一汽项目_试验室仿真技术简版资料.md"): section(
        f"""
        {HEADING}

        ### 1. 简版材料里的实验室仿真模块 -> 原型工程
        - 原文抓手：一页纸/两页纸材料中的“接口函数/节点库/实验室场景搭建”。
        - 工程对应：`{PROTO_BUILD}:47-68`；`{PROTO_BUILD}:107-182`；`{PROTO_RUN}:9-15`
        - 分析：这几行代码把简版材料里的模块划分真正搭成了 Simulink 工程，并保留最小参数入口用于快速复现实验。

        ### 2. 简版材料里的“智能生成与闭环验证” -> V7 工程主链
        - 原文抓手：`干扰特征库 + GA/GAN 生成最恶劣场景 + 闭环验证`。
        - 工程对应：`{SAT_MAIN}:281-341`；`{SAT_MAIN}:392-405`
        - 分析：主工程这里同时做最坏搜索、识别分类和关键帧导出，正好对应简版材料里最强调的“智能生成 + 验证闭环”。

        ### 3. 简版材料里的“评估与报告模板” -> 判定与摘要
        - 原文抓手：简版材料中的评分、报告模板和防护建议输出。
        - 工程对应：`{SAT_COMP}:8-53`；`{SAT_SUM}:10-38`
        - 分析：如果没有这两层，简版材料只能停留在方法宣讲；有了判定行表和摘要文本，简版材料才真正具备可交付性。
        """
    ),
    Path(r"D:\正向设计规范\obsidian知识库\20_参考节点\无人机研究_测试评估方法与实验室建设资料.md"): section(
        f"""
        {HEADING}

        ### 1. 测试评估方法 -> 功率域评估主链
        - 原文抓手：`无人机通信劣化-论文大纲_公式渲染版.docx` 中“路径损耗、遮挡判定、通信劣化指标”的方法段。
        - 工程对应：`{UAV_GA}:2412-2472`；`{UAV_GA}:2801-2865`
        - 分析：这些文档对规范线有价值，不是因为它们写成了论文，而是因为它们已经被 `UAV_GA.py` 实现成统一功率分解和路径损耗模块，可直接迁移“测试评估逻辑”。

        ### 2. 实验建设与语义布设 -> 场景约束层
        - 原文抓手：实验室建设方案和测试梳理里对场景、位置和热点布设的说明。
        - 工程对应：`{UAV_GA}:1101-1180`；`{UAV_GA}:1484-1565`
        - 分析：楼层面积统计与语义层投影正是把实验环境文本要求转成可复用场景约束的地方，所以这批建设文档对于规范线更像“测试场景构造规范”的素材。

        ### 3. 实测坏点与最坏工况 -> 对照验证链
        - 原文抓手：资料中关于热点对齐、最差点搜索和实测对照的部分。
        - 工程对应：`{UAV_MEAS}:530-640`；`{UAV_MEAS}:1186-1305`
        - 分析：新增无人机资料里最值得迁移到规范线的不是具体参数，而是“如何把公开测量点、热点和最坏搜索放到同一评估流程里”的方法学，这正由 `compare_measured_cornercases.py` 实现。

        ### 4. 结果输出与指标统计
        - 原文抓手：报告中的 BLER、吞吐、可靠性和能效输出口径。
        - 工程对应：`{UAV_EVAL}:243-328`；`{UAV_KPI}:58-123`
        - 分析：规范线若要吸收无人机测试评估方法，最终还是要吸收到“输出什么指标、怎样统计”的层面，这两段代码正对应这一层。
        """
    ),
    Path(r"D:\专利一汽\obsidian知识库\10_成果节点\一汽项目_试验室仿真支撑简版资料.md"): section(
        f"""
        {HEADING}

        ### 1. “实测-建模-仿真-智能生成-验证” -> 专利支撑主链
        - 原文句子：一页纸中对 `实测-建模-仿真-智能生成-验证闭环` 的描述。
        - 工程对应：`{PROTO_BUILD}:47-68`；`{PROTO_BUILD}:107-182`；`{SAT_MAIN}:281-341`
        - 分析：专利线需要证明测试系统方法不是纸上方案，而是已有原型与 V7 主工程共同支撑的，所以简版材料中的闭环描述正对应原型搭建和最坏工况搜索两段代码。

        ### 2. “EMC/通信联合评估” -> 判定表
        - 原文句子：`形成通信性能与 EMC 联合评估`。
        - 工程对应：`{SAT_COMP}:8-53`
        - 分析：专利文本若要落到“测试系统和方法”，就必须能把评估条目结构化输出，而这正由合规行表函数负责。

        ### 3. “评估报告与结论页” -> 摘要输出
        - 原文句子：材料里的 `输出评估报告模板 / 防护建议`。
        - 工程对应：`{SAT_SUM}:10-38`；`{SAT_MAIN}:853-854`
        - 分析：摘要输出文本就是专利交底书中“系统输出页”“评价结论页”的最近工程来源，所以这批简版材料与专利线的联系主要落在交付层而不是算法层。
        """
    ),
    Path(r"D:\专利一汽\obsidian知识库\10_成果节点\一汽项目_专利图示素材与模板.md"): section(
        f"""
        {HEADING}

        ### 1. 模板文档里的章节结构 -> 摘要与输出页
        - 原文抓手：`模板.docx` 中“项目概述 / 项目目标 / 研究方案”的结构，以及 `新建 DOCX 文档.docx` 里的研究报告版式。
        - 工程对应：`{SAT_SUM}:10-38`
        - 分析：这些模板最有价值的地方不是正文内容，而是它们与 `summary_v7.txt` 的输出结构天然一致，适合作为专利交底书、支撑报告和汇报页的版式母板。

        ### 2. 图示素材 -> 工程流程图的可视化锚点
        - 原文抓手：三张图片素材与文档中对系统流程图、模块图的占位。
        - 工程对应：`{PROTO_BUILD}:74-205`；`{SAT_MAIN}:281-341`
        - 分析：专利附图通常需要把“信号发射-干扰注入-接收补偿-评估输出”拆成模块图，这正好对应 Simulink 原型构建和 V7 最坏搜索主链，因此这些素材适合服务流程图和框图表达。

        ### 3. 研究报告草稿 -> 工程能力边界
        - 原文抓手：`新建 DOCX 文档.docx` 中提到的 `LEO StarNet EMC / GA on GAN / R2021a`。
        - 工程对应：`{SAT_MAIN}:1-40`；`{SAT_BUILD}:1-35`；`{SAT_SIM}:1-90`
        - 分析：该草稿里列举的平台能力，本质上可以回溯到 V7 工程入口、链路模型打包和时序仿真三层，所以图示与模板节点实际上是把已有工程能力重新包装成专利表达。

        ### 4. 图像导出来源
        - 原文抓手：需要放进专利或汇报中的关键帧、示意图、结果图。
        - 工程对应：`{SAT_MAIN}:392-405`
        - 分析：主工程在分类阶段可导出关键帧 STFT 图，因此专利图示里若出现“干扰识别/典型场景”图，最直接的工程来源就是这段导出逻辑，而不是手工二次画图。
        """
    ),
    Path(r"D:\专利一汽\obsidian知识库\20_参考节点\一汽项目_专利与标准参考合集.md"): section(
        f"""
        {HEADING}

        ### 1. 标准和专利边界 -> 统一配置层
        - 原文抓手：合集里的车载卫星通信专利、标准和研究方案定稿。
        - 工程对应：`{SAT_CFG}:120-163`
        - 分析：这批资料对专利线最核心的作用，是把哪些指标要考核、哪些转换量待后续提供、哪些输出文件要生成先固定到 `cfg.Requirements / cfg.Measurement / cfg.Output`。

        ### 2. “测试方法是否合规” -> 行表判定
        - 原文抓手：参考合集里的标准条文和现有方案比较。
        - 工程对应：`{SAT_COMP}:8-53`
        - 分析：如果没有这层判定表，专利线只能写“能够评估”；有了这层，专利线可以进一步写“能够依据某些指标自动给出是否满足要求”的系统方法。

        ### 3. “系统输出页/结果结论” -> 摘要文本
        - 原文抓手：资料中的结果说明和标准化结论表述。
        - 工程对应：`{SAT_SUM}:10-38`
        - 分析：这让参考资料不只停留在背景技术，还能反向约束专利交底书里系统输出页应该怎样写、怎样体现最坏工况与基线工况差异。
        """
    ),
    Path(r"D:\专利自有\obsidian知识库\10_成果节点\一汽项目_卫星EMC原型工程与数据集.md"): section(
        f"""
        {HEADING}

        ### 1. 本地复制的原型工程 -> 自动建模主入口
        - 原文抓手：该节点内复制的 `LEO_EMC_Sim` 和根目录 `build_LEO_EMC_Sim.m`。
        - 工程对应：`{OWN_BUILD}:1-20`；`{OWN_BUILD}:47-68`；`{OWN_BUILD}:107-182`
        - 分析：专利自有线里的这批原型资产并非备份，而是保留了自动搭建 Simulink 通信链、配置物理层/干扰层和输出指标层的最小实现。

        ### 2. 本地复制的干扰与补偿算法
        - 原文抓手：节点内 `cfo_comp.m`、`interf_gen.m`、`run_LEO_EMC_Sim.m`、`preamble_insert.m`。
        - 工程对应：`{OWN_INTF}:1-37`；`{OWN_CFO}:1-23`；`{OWN_RUN}:1-15`；`{OWN_PRE}:1-25`
        - 分析：这几份文件构成了专利自有线最值得继承的“干扰注入-接收补偿-参数入口-帧结构”四件套，是以后定义第二个专利实施例时最容易被复用的代码块。

        ### 3. 本地复制的 V1 链路模型 -> 物理层证据链
        - 原文抓手：节点内 `LEO_Sim_V1.m`。
        - 工程对应：`{OWN_V1}:123-178`；`{OWN_V1}:229-321`；`{OWN_V1}:585-634`
        - 分析：V1 代码把场景、C/N/I、干扰波形和蒙特卡罗链路评价明确写成了可复现实验，是专利自有线提炼“测试方法/系统功能模块”时最底层的物理证据。

        ### 4. 数据集与训练模型在工程中的真实位置
        - 原文抓手：该节点同时包含 `dataset_stft_r2021a`、`GAN_Jammer_R2021a.mat`、`lenet_stft_model_r2021a.mat`。
        - 工程对应：`{SAT_DATA}:1-20`；`{SAT_TRAIN}:1-45`；`{SAT_CLS}:1-84`
        - 分析：专利自有节点里虽然复制了数据资产，但这些资产的生成、训练和在线使用逻辑仍然在卫星主工程侧完成，因此它们在本节点里的意义是“可继承训练底座”，而不是独立算法实现。
        """
    ),
}


def upsert_section(path: Path, heading: str, content: str) -> None:
    text = path.read_text(encoding="utf-8")
    marker = f"\n{heading}\n"
    if marker in text:
        prefix = text.split(marker, 1)[0].rstrip()
        text = prefix + "\n\n" + content.strip() + "\n"
    else:
        text = text.rstrip() + "\n\n" + content.strip() + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> int:
    for path, content in SECTIONS.items():
        if not path.exists():
            print(f"[SKIP] {path}")
            continue
        upsert_section(path, HEADING, content)
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
