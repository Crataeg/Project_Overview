# UAV Communication GA Project Guide

## Active Files

- `UAV_GA.py`：核心场景生成、链路预算、margin 劣化评估、GA 问题定义
- `gan_uav_pipeline.py`：GAN 训练与样本生成
- `compare_random_ga_gan.py`：正式主流程；输出 `GA vs GAN vs Random` 对比、KPI、可视化，并在本地有测量数据时自动追加测量 corner-case 对比
- `evaluate.py`：独立 KPI/BLER/吞吐评估入口；可选追加测量 corner-case 对比
- `kpi.py`：`outage / rate / EE / tail risk` 统计
- `bler_mc.py`：BLER_A（Monte Carlo）
- `bler_sionna.py`：BLER_B（Sionna LUT）
- `plot_results.py`：KPI 出图
- `compare_measured_cornercases.py`：测量轨迹点 vs 模型最劣区域 vs 随机点，对齐 AERPAW / Dryad 环境并输出空间热力图
- `make_images_docx.py`：报告整理工具

## Archive / Reference

- `__backup_before_upstream_20260302_143119/`：历史备份，不参与当前流程
- `__upstream_UAV_GA_GAN/`：上游原始参考，不参与当前流程
- `key reference/`：论文与参考资料

## Recommended Workflow

1. **主环境**
   - 日常运行：`.\.venv\Scripts\python.exe`
   - Sionna 仅用于 `BLER_B`：`.\.venv_sionna\Scripts\python.exe`

2. **生成 GA / GAN 样本**
   - 跑 `gan_uav_pipeline.py` 或你的既有训练入口
   - 目标输出：`output/<run_id>/gan/arrays/ga_samples.npy` 和 `gan_samples.npy`

3. **正式实验**
   - 推荐入口：`compare_random_ga_gan.py`
   - 输出目录：`output/<run_id>/compare/`
   - 自动产物：
     - `comparison_summary.json`
     - `kpi/`
     - `visuals/`
     - `measured_cornercases/`（当本地测量 zip 存在且未显式跳过）

4. **独立 KPI 复跑**
   - 用 `evaluate.py`
   - 适合在不重跑 GA/GAN 的情况下补 `BLER_A / BLER_B / measured corner-case`

## Output Layout

- `output/<run_id>/gan/arrays/`：样本数组
- `output/<run_id>/compare/`：主对比输出
- `output/<run_id>/compare/kpi/`：独立 KPI / BLER / 吞吐
- `output/<run_id>/compare/measured_cornercases/`：AERPAW / Dryad 测量对照
  - `selected_points_with_source.csv`
  - `trajectory_points_modeled.csv`
  - `grid_map.csv`
  - `*_spatial_compare.png`
  - `*_dense_heatmap.png`
  - `corner_case_report.json`

## Common Commands

- 正式主流程：
  - `& .\.venv\Scripts\python.exe .\compare_random_ga_gan.py --run_id <run_id>`
- 正式主流程 + 禁用测量对照：
  - `& .\.venv\Scripts\python.exe .\compare_random_ga_gan.py --run_id <run_id> --skip_measured_corner_compare`
- 独立 KPI：
  - `& .\.venv\Scripts\python.exe .\evaluate.py --run_id <run_id>`
- 独立 KPI + Sionna LUT：
  - `& .\.venv\Scripts\python.exe .\evaluate.py --run_id <run_id> --gen_bler_b --sionna_python .\.venv_sionna\Scripts\python.exe`
- 独立测量对照：
  - `& .\.venv\Scripts\python.exe .\compare_measured_cornercases.py`

## Maintenance Rules

- 改链路预算或 margin 公式时，同时检查：
  - `UAV_GA.py`
  - `evaluate.py`
  - `compare_measured_cornercases.py`
- 改 KPI 定义时，只改：
  - `kpi.py`
  - `plot_results.py`
- 改 BLER 曲线时：
  - `bler_mc.py`
  - `bler_sionna.py`
  - `evaluate.py`
- 测量数据 zip 保持在外部路径，不要拷进仓库
- 清理临时输出可运行：
  - `powershell -ExecutionPolicy Bypass -File .\clean_outputs.ps1`

## What To Keep

- 保留：上面 `Active Files`、`.venv`、`.venv_sionna`、`output/<run_id>/`
- 可清理：`output/*smoke*`、`output/__tmp_*`、临时 BLER csv、`__pycache__/`
- 仅归档：`__backup_before_*`、`__upstream_*`
