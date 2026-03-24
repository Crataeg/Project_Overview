from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path


ROOT_D = Path("D:/")
OVERVIEW_ROOT = Path(r"D:\工程总览")
TOTAL_VAULT = OVERVIEW_ROOT / "总的obsidian知识库"

PROJECTS = {
    "专利自有": ROOT_D / "专利自有" / "LEO_Sim_Patent",
    "专利一汽": ROOT_D / "专利一汽" / "LEO_EMCSim_Lab",
    "论文无人机": ROOT_D / "论文无人机" / "UAV_GA_GAN",
    "论文卫星": ROOT_D / "论文卫星" / "LEO_Sim",
    "正向设计规范": ROOT_D / "正向设计规范" / "Forward_Design",
}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel_link(from_path: Path, to_path: Path) -> str:
    return os.path.relpath(to_path, from_path.parent).replace("\\", "/")


def force_remove(path: Path) -> None:
    def _onexc(func, target, _excinfo):
        try:
            os.chmod(target, stat.S_IWRITE)
        except Exception:
            pass
        func(target)

    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, onexc=_onexc)
    else:
        try:
            os.chmod(path, stat.S_IWRITE)
        except Exception:
            pass
        path.unlink()


def reset_dir(path: Path) -> None:
    if path.exists():
        force_remove(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        force_remove(dst)
    shutil.copytree(src, dst)


def copy_files(src_dir: Path, dst_dir: Path, names: list[str]) -> list[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in names:
        src = src_dir / name
        if src.exists():
            dst = dst_dir / name
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def append_section(path: Path, marker: str, text: str) -> None:
    content = path.read_text(encoding="utf-8")
    if marker in content:
        return
    updated = content.rstrip() + "\n\n" + text.rstrip() + "\n"
    path.write_text(updated, encoding="utf-8")


def build_output_folders() -> dict[str, Path]:
    outputs: dict[str, Path] = {}

    uav_output = PROJECTS["论文无人机"] / "产出文件夹" / "无人机仿真平台工程文件夹_最终版"
    reset_dir(uav_output)
    copy_dir(
        PROJECTS["论文无人机"] / "成果本身" / "代码工程" / "12",
        uav_output / "12",
    )
    write_text(
        uav_output / "README.md",
        """
# 无人机仿真平台工程文件夹_最终版

- 来源：`D:\\论文无人机\\UAV_GA_GAN\\成果本身\\代码工程\\12`
- 定位：Python 无人机通信性能评估平台最终工程归档
- 关键主链：`UAV_GA.py -> gan_uav_pipeline.py -> compare_random_ga_gan.py -> evaluate.py -> kpi.py`
- 关键输出：`output\\measured_corner_compare\\corner_case_summary.json`
""",
    )
    outputs["论文无人机"] = uav_output

    sat_output = PROJECTS["论文卫星"] / "产出文件夹" / "卫星仿真平台工程文件夹_最终版"
    reset_dir(sat_output)
    copy_dir(
        PROJECTS["论文卫星"] / "成果本身" / "代码工程" / "LEO_Sim",
        sat_output / "LEO_Sim",
    )
    write_text(
        sat_output / "README.md",
        """
# 卫星仿真平台工程文件夹_最终版

- 来源：`D:\\论文卫星\\LEO_Sim\\成果本身\\代码工程\\LEO_Sim`
- 定位：MATLAB 低轨卫星 EMC 平台最终工程归档
- 关键主链：`LEO_StarNet_EMC_V7_0_Engineering.m -> emcBuildLinkModel.m -> simulateStarNetV7.m -> emcCombineE2E.m`
- 干扰增强链：`trainOrLoadJammerGAN.m -> worstCaseObjectiveV7.m`
- 识别链：`generateDatasetSimpleSTFT.m -> trainLeNetSTFT.m -> classifyInterferenceTimeline_powerSampler.m`
""",
    )
    outputs["论文卫星"] = sat_output

    patent_src = PROJECTS["专利一汽"] / "成果本身" / "专利文本与附图" / "专利一汽"
    patent_output = PROJECTS["专利一汽"] / "产出文件夹" / "专利一汽_交底书与附图最终版"
    reset_dir(patent_output)
    copy_files(
        patent_src,
        patent_output,
        [
            "20260318_交底书_第三版_一种卫星通信天线的实验室环境模拟测试系统及方法.docx",
            "20260318_附图清单_第三版_卫星通信天线的实验室环境模拟测试系统_重新生成可下载版.docx",
            "20260318_附图清单_优化重绘版_可下载.docx",
            "20260318_Visio附图_卫星通信天线实验室环境模拟测试系统_10图_SVG_PNG.zip",
            "20260318_专利附图_优化重绘版_SVG_PNG.zip",
        ],
    )
    write_text(
        patent_output / "README.md",
        """
# 专利一汽_交底书与附图最终版

- 来源：`D:\\专利一汽\\LEO_EMCSim_Lab\\成果本身\\专利文本与附图\\专利一汽`
- 交付对象：一汽专利交底书、附图清单、附图压缩包
- 技术支撑来源：卫星 EMC 仿真平台、试验室环境仿真报告、专利与标准对照材料
""",
    )
    outputs["专利一汽"] = patent_output

    own_patent_output = PROJECTS["专利自有"] / "产出文件夹" / "专利自有_交底书待补位"
    reset_dir(own_patent_output)
    template_src = patent_src / "20250912_交底书模板_发明及实用新型.docx"
    if template_src.exists():
        shutil.copy2(template_src, own_patent_output / template_src.name)
    write_text(
        own_patent_output / "README.md",
        """
# 专利自有_交底书待补位

- 当前状态：尚未发现成型交底书，本目录作为第二个专利的正式产出位保留
- 已放入资料：交底书模板
- 建议继承链：`D:\\专利自有\\LEO_Sim_Patent\\成果本身\\代码工程\\LEO_Sim` 的链路/干扰/最坏工况能力
- 建议补齐内容：发明点定义、实施例分层、附图编号、参数换算口径
""",
    )
    outputs["专利自有"] = own_patent_output

    design_src = PROJECTS["正向设计规范"] / "成果本身" / "规范方案" / "车载低轨卫星通信系统EMC性能正向设计技术研究方案"
    design_output = PROJECTS["正向设计规范"] / "产出文件夹" / "正向设计规范_最终稿"
    reset_dir(design_output)
    copy_files(
        design_src,
        design_output,
        [
            "最终稿_车载低轨卫星通信系统EMC性能正向设计技术研究方案.docx",
            "Q_FC_XXX-2025_车载低轨卫星通信系统EMC性能正向设计规范.docx",
            "正向设计规范0312.docx",
            "车载低轨卫星通信系统EMC正向设计规范_专利与标准对照文件.docx",
        ],
    )
    write_text(
        design_output / "README.md",
        """
# 正向设计规范_最终稿

- 来源：`D:\\正向设计规范\\Forward_Design\\成果本身\\规范方案\\车载低轨卫星通信系统EMC性能正向设计技术研究方案`
- 交付对象：正向设计研究方案、规范草案、专利与标准对照稿
- 技术落脚点：将卫星平台配置、指标阈值、合规判据固化为规范语言
""",
    )
    outputs["正向设计规范"] = design_output

    total_output = OVERVIEW_ROOT / "产出文件夹"
    reset_dir(total_output)
    for name, src in outputs.items():
        copy_dir(src, total_output / name)
    write_text(
        total_output / "README.md",
        """
# 工程总览产出文件夹

本目录集中保存五条主线中当前最适合作为“最终交付包”的版本：

- `论文无人机`：无人机仿真平台工程文件夹最终版
- `论文卫星`：卫星仿真平台工程文件夹最终版
- `专利一汽`：一汽专利交底书与附图最终版
- `专利自有`：自有专利交底书待补位
- `正向设计规范`：正向设计规范最终稿
""",
    )
    outputs["总览"] = total_output
    return outputs


def build_project_notes(outputs: dict[str, Path]) -> None:
    uav_vault = PROJECTS["论文无人机"] / "obsidian知识库"
    sat_vault = PROJECTS["论文卫星"] / "obsidian知识库"
    patent_faw_vault = PROJECTS["专利一汽"] / "obsidian知识库"
    patent_own_vault = PROJECTS["专利自有"] / "obsidian知识库"
    design_vault = PROJECTS["正向设计规范"] / "obsidian知识库"

    uav_code = uav_vault / "40_代码级技术解析" / "无人机仿真平台_代码级技术解析.md"
    sat_code = sat_vault / "40_代码级技术解析" / "卫星仿真平台_代码级技术解析.md"
    patent_faw_code = patent_faw_vault / "40_代码级技术解析" / "专利一汽_技术方案与仿真支撑.md"
    patent_own_code = patent_own_vault / "40_代码级技术解析" / "专利自有_继承基础与待补项.md"
    design_code = design_vault / "40_代码级技术解析" / "正向设计规范_代码与规范映射.md"

    write_text(
        uav_code,
        f"""
# 无人机仿真平台_代码级技术解析

## 代码主链

- `D:\\论文无人机\\UAV_GA_GAN\\成果本身\\代码工程\\12\\UAV_GA.py`
  - `InterferenceSourceConfig / Building / InterferenceSource` 负责把城市环境、建筑、干扰体类型和参数离散化。
  - `DroneCommProblem(ea.Problem)` 是整个优化问题的核心封装，承担变量边界、场景生成、目标函数和约束评价。
  - `_compute_building_observables()` 把建筑统计特征变成可复用的遮挡/观测缓存，降低 GA 重复评估噪声。
  - `_precompute_interference_power_mw_per_drone()` 在场景固定后先把每架无人机看到的干扰功率预展开，避免每次评分都重复全量几何计算。
  - `compute_sinr()` 与 `compute_sinr_zeng()` 分别给出链路质量计算口径，是后续 KPI/BLER 的底层输入。
- `D:\\论文无人机\\UAV_GA_GAN\\成果本身\\代码工程\\12\\gan_uav_pipeline.py`
  - `run_ga_samples()` 先从 `DroneCommProblem` 采样出高退化场景。
  - `MLPGenerator / MLPDiscriminator` 把最坏场景分布学习成可生成模型，不再只依赖 GA 重跑。
  - `train_gan()` 与 `generate_gan_samples()` 负责把 GA 样本扩展为 GAN 极端样本集。
- `D:\\论文无人机\\UAV_GA_GAN\\成果本身\\代码工程\\12\\compare_random_ga_gan.py`
  - `evaluate_samples()` 统一对 `GA / GAN / Random` 三组样本做同口径评分。
  - `plot_comparison()` 和 `plot_total_deg_focus()` 给出退化分布对比，把“极端场景是否真的更坏”做成可视化证据。
- `D:\\论文无人机\\UAV_GA_GAN\\成果本身\\代码工程\\12\\evaluate.py`
  - `extract_link_records()` 把场景样本展开成链路记录。
  - `evaluate_groups_from_samples()` 统一生成 `BLER_A / BLER_B / throughput / outage / KPI` 输出，是论文结果层的主入口。
- `D:\\论文无人机\\UAV_GA_GAN\\成果本身\\代码工程\\12\\kpi.py`
  - `compute_kpis()` 汇总 SINR/outage 等统计量。
  - `bler_interpolate()` 与 `compute_throughput_kpis()` 完成 BLER 到吞吐的二级映射。

## 技术闭环

1. `UAV_GA.py` 定义城市、无人机、干扰源和评分函数。
2. `gan_uav_pipeline.py` 从 GA 最坏样本学习 GAN 分布，扩展极端样本空间。
3. `compare_random_ga_gan.py` 验证 GA/GAN 相对随机场景的退化提升。
4. `evaluate.py + kpi.py` 把退化场景转成论文能直接引用的 BLER/KPI/吞吐指标。
5. `output\\measured_corner_compare\\corner_case_summary.json` 再把模型最坏区域和公开实测坏点对齐。

## 与文献/规范/专利的关系

- 无人机线自身不直接服务专利交底，但它提供了“最坏工况搜索 + 指标层评价”的方法论，可迁移到卫星 EMC 平台。
- 它和卫星线的共通点不是载体，而是“先建模，再最坏搜索，再指标验证”的工程方法。

## 产出文件夹

- 最终版路径：`{outputs["论文无人机"]}`
- 总览镜像：`{outputs["总览"] / "论文无人机"}`
""",
    )

    write_text(
        sat_code,
        f"""
# 卫星仿真平台_代码级技术解析

## 代码主链

- `D:\\论文卫星\\LEO_Sim\\成果本身\\代码工程\\LEO_Sim\\v7proj\\LEO_StarNet_EMC_V7_0_Engineering.m`
  - 主控入口。负责装配配置、建立上下行模型、调用基线仿真、最坏工况搜索、分类识别和结果汇总。
  - `emcDefaultConfig()` 提供需求阈值、链路参数、干扰机参数与可视化开关。
  - `emcBuildLinkModel('DL'/'UL', ...)` 把几何、功率、干扰和时间轴压成上下行链路模型。
  - `simulateStarNetV7()` 对基线场景和最坏场景分别求解，输出 `SINR / THR / outageFrac`。
  - `emcCombineE2E()` 用 `min(DL, UL)` 形成端到端吞吐和中断判据。
  - `trainOrLoadJammerGAN()` 与 `worstCaseObjectiveV7()` 构成“干扰生成 + 最坏工况搜索”闭环。
  - `classifyInterferenceTimeline_powerSampler()` 把链路功率时序映射到 STFT 图像，再做干扰类型识别。
- `D:\\论文卫星\\LEO_Sim\\成果本身\\代码工程\\LEO_Sim\\v7proj\\emcDefaultConfig.m`
  - 规范化保存需求阈值、平台参数、干扰源参数和合规判据，是平台和规范的接口层。
- `D:\\论文卫星\\LEO_Sim\\成果本身\\代码工程\\LEO_Sim\\v7proj\\emcBuildLinkModel.m`
  - 链路建模层，把卫星/终端几何、路径损耗、自干扰和外部干扰封装成求解输入。
- `D:\\论文卫星\\LEO_Sim\\成果本身\\代码工程\\LEO_Sim\\v7proj\\simulateStarNetV7.m`
  - 求解层，把链路模型和干扰场景压缩成时序 `SINR / THR / outageFrac`。
- `D:\\论文卫星\\LEO_Sim\\成果本身\\代码工程\\LEO_Sim\\v7proj\\emcComputeComplianceRowsV7.m`
  - 合规层，把仿真结果转成“指标名称-目标值-是否通过”的行表结构。
- `D:\\论文卫星\\LEO_Sim\\成果本身\\代码工程\\LEO_Sim\\v7proj\\emcWriteSummaryTextV7.m`
  - 交付层，把关键结果写成摘要文本，供报告、规范和专利输出页直接复用。

## 两条算法支链

### 最坏工况搜索支链

- `genJammerSimple.m` 生成基础干扰体。
- `trainOrLoadJammerGAN.m` 学习复杂干扰时序。
- `genJamAggFromG.m` 把生成器输出映射回时域干扰聚合量。
- `worstCaseObjectiveV7.m` 以吞吐/中断为目标进行最坏搜索。

### 干扰识别支链

- `generateDatasetSimpleSTFT.m` 生成 STFT 数据集。
- `trainLeNetSTFT.m` 训练图像分类网络。
- `classifyInterferenceTimeline_powerSampler.m` 对最坏时序做在线分类。
- `sampleIQFromPowers.m` 把功率级输入还原成接收 IQ 片段，是识别支链和物理层之间的转换桥。

## 与专利/规范的关系

- 专利一汽里的“参数扫描、代表场强提取、等效输入换算、输出报告页面”都能在 `emcBuildLinkModel / simulateStarNetV7 / worstCaseObjectiveV7 / emcWriteSummaryTextV7` 这条链上找到代码映射。
- 正向设计规范里的阈值条款和合规性表格，可以直接落到 `emcDefaultConfig / emcComputeComplianceRowsV7`。

## 产出文件夹

- 最终版路径：`{outputs["论文卫星"]}`
- 总览镜像：`{outputs["总览"] / "论文卫星"}`
""",
    )

    write_text(
        patent_faw_code,
        f"""
# 专利一汽_技术方案与仿真支撑

## 当前产出

- 交底书：`20260318_交底书_第三版_一种卫星通信天线的实验室环境模拟测试系统及方法.docx`
- 附图清单：`20260318_附图清单_第三版_卫星通信天线的实验室环境模拟测试系统_重新生成可下载版.docx`
- 附图压缩包：`20260318_Visio附图_卫星通信天线实验室环境模拟测试系统_10图_SVG_PNG.zip`
- 优化重绘包：`20260318_专利附图_优化重绘版_SVG_PNG.zip`

## 对应的代码支撑

- 图中的“系统总体架构 / 测试总体布局 / 位置关系 / 空间采样区域”对应卫星平台的 `emcBuildLinkModel.m` 和几何/链路参数装配层。
- 图中的“代表场强提取 / 场强到接收机等效输入信号换算”对应 `simulateStarNetV7.m` 输出的链路功率、SINR 与门限判据处理。
- 图中的“灵敏度判定 / 参数扫描与最不利工况输出”对应 `worstCaseObjectiveV7.m`、`emcComputeComplianceRowsV7.m` 和 `emcWriteSummaryTextV7.m`。
- 图中的“输出报告页面示意图”本质上是把 `summary_v7.txt` 与合规结果行表做界面化表达。

## 技术关系

- 这份专利不是独立算法工程，而是把卫星 EMC 平台的链路求解、参数换算、最坏搜索和结果输出固化成可申请的测试系统方法。
- 因此它和 `论文卫星` 共用仿真底座，和 `正向设计规范` 共用指标口径，和 `专利自有` 共用专利表达模板与对标方式。

## 产出文件夹

- 最终版路径：`{outputs["专利一汽"]}`
- 总览镜像：`{outputs["总览"] / "专利一汽"}`
""",
    )

    write_text(
        patent_own_code,
        f"""
# 专利自有_继承基础与待补项

## 当前状态

- 当前电脑上未发现第二个专利的成型交底书，因此本项目先以“待补专利产出位”存在。
- 现阶段已具备可直接继承的底座：卫星 EMC 仿真平台、主机厂专利对标资料、专利交底模板、正向设计规范。

## 可继承的代码基础

- `D:\\专利自有\\LEO_Sim_Patent\\成果本身\\代码工程\\LEO_Sim` 提供完整的链路建模、最坏工况搜索与结果输出能力。
- `D:\\专利自有\\LEO_Sim_Patent\\成果本身\\代码工程\\satellite.m` 作为兼容入口，保留基础链路脚本位。
- `D:\\专利自有\\LEO_Sim_Patent\\参考文献\\专利对标` 中的一汽及多主机厂资料可用于界定创新边界。

## 建议补齐顺序

1. 先定义发明点到底落在“测试系统”“参数换算方法”还是“抗干扰策略”。
2. 再决定需要继承卫星平台的哪一层代码能力。
3. 最后再生成交底书正文、附图清单和实施例。

## 产出文件夹

- 待补位路径：`{outputs["专利自有"]}`
- 总览镜像：`{outputs["总览"] / "专利自有"}`
""",
    )

    write_text(
        design_code,
        f"""
# 正向设计规范_代码与规范映射

## 规范如何落到代码

- `Q_FC_XXX-2025_车载低轨卫星通信系统EMC性能正向设计规范.docx`
  - 对应平台中的需求阈值、判据和场景约束。
  - 代码落点：`emcDefaultConfig.m`。
- `车载低轨卫星通信系统EMC正向设计规范_专利与标准对照文件.docx`
  - 用于把规范条文与专利表达统一口径。
  - 代码落点：`emcComputeComplianceRowsV7.m` 与 `emcWriteSummaryTextV7.m` 形成“仿真结果 -> 规范条款”映射。
- `最终稿_车载低轨卫星通信系统EMC性能正向设计技术研究方案.docx`
  - 负责把工程目标、测试对象、评价流程和输出指标组织成方案语言。
  - 代码落点：`LEO_StarNet_EMC_V7_0_Engineering.m` 主流程。

## 规范化闭环

1. `emcDefaultConfig.m` 定义输入约束与目标值。
2. `LEO_StarNet_EMC_V7_0_Engineering.m` 组织基线场景和最坏场景求解。
3. `emcComputeComplianceRowsV7.m` 生成是否满足阈值的判定表。
4. `emcWriteSummaryTextV7.m` 把判定表和摘要输出成报告口径。
5. 规范/研究方案文档再把这套口径转写成条文、模板和交付要求。

## 与专利、论文的关系

- 对 `论文卫星`：提供需求来源和讨论框架。
- 对 `专利一汽`：提供专利与标准对照口径。
- 对 `专利自有`：提供未来交底书的指标和术语骨架。

## 产出文件夹

- 最终版路径：`{outputs["正向设计规范"]}`
- 总览镜像：`{outputs["总览"] / "正向设计规范"}`
""",
    )

    append_section(
        uav_vault / "00_项目总览.md",
        "## 代码级技术解析",
        f"""
## 代码级技术解析

- [无人机仿真平台_代码级技术解析]({rel_link(uav_vault / '00_项目总览.md', uav_code)})

## 产出文件夹

- `D:\\论文无人机\\UAV_GA_GAN\\产出文件夹`
- 总览镜像：`D:\\工程总览\\产出文件夹\\论文无人机`
""",
    )
    append_section(
        sat_vault / "00_项目总览.md",
        "## 代码级技术解析",
        f"""
## 代码级技术解析

- [卫星仿真平台_代码级技术解析]({rel_link(sat_vault / '00_项目总览.md', sat_code)})

## 产出文件夹

- `D:\\论文卫星\\LEO_Sim\\产出文件夹`
- 总览镜像：`D:\\工程总览\\产出文件夹\\论文卫星`
""",
    )
    append_section(
        patent_faw_vault / "00_项目总览.md",
        "## 代码级技术解析",
        f"""
## 代码级技术解析

- [专利一汽_技术方案与仿真支撑]({rel_link(patent_faw_vault / '00_项目总览.md', patent_faw_code)})

## 产出文件夹

- `D:\\专利一汽\\LEO_EMCSim_Lab\\产出文件夹`
- 总览镜像：`D:\\工程总览\\产出文件夹\\专利一汽`
""",
    )
    append_section(
        patent_own_vault / "00_项目总览.md",
        "## 代码级技术解析",
        f"""
## 代码级技术解析

- [专利自有_继承基础与待补项]({rel_link(patent_own_vault / '00_项目总览.md', patent_own_code)})

## 产出文件夹

- `D:\\专利自有\\LEO_Sim_Patent\\产出文件夹`
- 总览镜像：`D:\\工程总览\\产出文件夹\\专利自有`
""",
    )
    append_section(
        design_vault / "00_项目总览.md",
        "## 代码级技术解析",
        f"""
## 代码级技术解析

- [正向设计规范_代码与规范映射]({rel_link(design_vault / '00_项目总览.md', design_code)})

## 产出文件夹

- `D:\\正向设计规范\\Forward_Design\\产出文件夹`
- 总览镜像：`D:\\工程总览\\产出文件夹\\正向设计规范`
""",
    )


def build_total_notes(outputs: dict[str, Path]) -> None:
    code_overview = TOTAL_VAULT / "20_交叉图谱" / "代码级技术总览.md"
    out_nav = TOTAL_VAULT / "20_交叉图谱" / "产出文件夹导航.md"

    write_text(
        code_overview,
        f"""
# 代码级技术总览

## 两个平台

- [论文无人机代码解析]({rel_link(code_overview, PROJECTS["论文无人机"] / "obsidian知识库" / "40_代码级技术解析" / "无人机仿真平台_代码级技术解析.md")})
- [论文卫星代码解析]({rel_link(code_overview, PROJECTS["论文卫星"] / "obsidian知识库" / "40_代码级技术解析" / "卫星仿真平台_代码级技术解析.md")})

## 两个专利

- [专利一汽技术支撑]({rel_link(code_overview, PROJECTS["专利一汽"] / "obsidian知识库" / "40_代码级技术解析" / "专利一汽_技术方案与仿真支撑.md")})
- [专利自有待补与继承链]({rel_link(code_overview, PROJECTS["专利自有"] / "obsidian知识库" / "40_代码级技术解析" / "专利自有_继承基础与待补项.md")})

## 正向设计规范

- [规范与代码映射]({rel_link(code_overview, PROJECTS["正向设计规范"] / "obsidian知识库" / "40_代码级技术解析" / "正向设计规范_代码与规范映射.md")})

## 统一关系

- `论文无人机` 提供“最坏工况搜索 + 指标评价”方法链。
- `论文卫星` 提供“链路建模 + 最坏搜索 + 合规判定 + 识别分类”工程链。
- `专利一汽` 把卫星平台的链路求解和判定流程固化成专利表达。
- `专利自有` 当前保留第二专利的产出位，待在现有平台能力基础上补齐发明点。
- `正向设计规范` 把平台配置、阈值和合规判据固化为规范文本。
""",
    )

    write_text(
        out_nav,
        f"""
# 产出文件夹导航

## 总览层

- `D:\\工程总览\\产出文件夹`

## 分项目

- `D:\\论文无人机\\UAV_GA_GAN\\产出文件夹`
- `D:\\论文卫星\\LEO_Sim\\产出文件夹`
- `D:\\专利一汽\\LEO_EMCSim_Lab\\产出文件夹`
- `D:\\专利自有\\LEO_Sim_Patent\\产出文件夹`
- `D:\\正向设计规范\\Forward_Design\\产出文件夹`

## 当前最终版

- 无人机仿真平台：`{outputs["论文无人机"]}`
- 卫星仿真平台：`{outputs["论文卫星"]}`
- 一汽专利交底书：`{outputs["专利一汽"]}`
- 自有专利待补位：`{outputs["专利自有"]}`
- 正向设计规范最终稿：`{outputs["正向设计规范"]}`
""",
    )

    append_section(
        TOTAL_VAULT / "00_工程总览.md",
        "## 代码级技术总览",
        f"""
## 代码级技术总览

- [代码级技术总览]({rel_link(TOTAL_VAULT / '00_工程总览.md', code_overview)})

## 产出文件夹导航

- [产出文件夹导航]({rel_link(TOTAL_VAULT / '00_工程总览.md', out_nav)})
- 总览产出目录：`D:\\工程总览\\产出文件夹`
""",
    )

    manifest = {
        "generated_at": str(Path(__file__).resolve()),
        "project_outputs": {name: str(path) for name, path in outputs.items()},
    }
    write_text(
        TOTAL_VAULT / "90_自动索引" / "postprocess_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2),
    )


def main() -> int:
    outputs = build_output_folders()
    build_project_notes(outputs)
    build_total_notes(outputs)
    print("Postprocess completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

