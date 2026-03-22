# 干扰源对照总表（代码事实）

来源：`UAV_GA.py`（当前工程实现）。本表按“干扰源类型”汇总：数据字段、生成/标定、activity、PPP 强度与空间混合、语义约束、传播附加项与关键代码入口。

> 口径说明（重要）：本工程中 `power_range(dBm)` 在链路预算里直接作为 `P_tx[dBm]` 使用（`P_rx = P_tx - PL + ...`），因此最稳妥的论文表述为 **EIRP‑equivalent dBm**（天线增益/馈线损耗等折算进这一项）。

## 1) 对照总表

| type_id | type_key | 名称 | 频率 `f` | `power_range(dBm)` 口径 | 功率/标定生成 | activity（平均占空） | 场景语义约束 Γ(type) | PPP λ 默认值（sources/km²） | PPP 位置混合（ground/rooftop/indoor） | 传播附加项（除 FSPL+LoS/NLoS 外） | 关键代码入口 |
|---:|---|---|---:|---|---|---|---|---:|---|---|---|
| 0 | `wifi_2_4g` | WiFi 2.4GHz AP | 2.4e9 | EIRP‑equivalent dBm（在链路里当 `P_tx`） | 下限裁剪 `eirp_min_dbm=5` | `traffic_duty + beacon_duty`（默认 0.15 + 0.5ms/102.4ms） | `['indoor']` | 60 | 0.10 / 0.10 / 0.80 | `PL += 10log10(path_loss_factor*height_factor)`；室内源可叠加 P.1238/P.2109 | 配置 `UAV_GA.py:59`；activity `UAV_GA.py:1643`；PPP `UAV_GA.py:1029`；传播 `UAV_GA.py:2096` |
| 1 | `wifi_5_8g` | WiFi 5.8GHz AP | 5.8e9 | 同上 | 下限裁剪 `eirp_min_dbm=8` | 同 Wi‑Fi | `['indoor']` | 40 | 0.10 / 0.10 / 0.80 | 同上 | 配置 `UAV_GA.py:80`；activity `UAV_GA.py:1643`；PPP `UAV_GA.py:1029` |
| 2 | `cellular_4g` | 4G 基站 | 2.6e9 |（当前未显式标注）但在链路里仍当 `P_tx` 使用 | 直接按区间采样/裁剪（无 profile 锚点） | 1.0 | `['roof','tower']` | 6 | 0.05 / 0.90 / 0.05 | 同构 A2G；无额外标定/占空 | 配置 `UAV_GA.py:101`；语义 `UAV_GA.py:235`；PPP `UAV_GA.py:1034` |
| 3 | `cellular_5g` | 5G 基站（按小站档位） | 3.5e9 |（未显式标注）但按 EIRP‑equivalent 使用 | femto/pico/micro 锚点（20/24/36 dBm）+ ±3 dB 偏移；`profile_key=5g_smallcell_*` | 固定 0.70 | `['roof','pole','tower']` | 8 | 0.10 / 0.85 / 0.05 | 同构 A2G | 配置 `UAV_GA.py:112`；标定 `UAV_GA.py:1738`；activity `UAV_GA.py:1643` |
| 4 | `gnss_jammer` | GNSS 干扰机 | 1.575e9 | `eirp_equivalent_dbm`（已写死） | `calibration.eirp_range_dbm=(20,40)`（带 sources 占位） | 1.0 | `['street','sidewalk']` | 1 | 0.90 / 0.05 / 0.05 | 同构 A2G；室内才会叠加 P.1238/P.2109 | 配置 `UAV_GA.py:164`；语义 `UAV_GA.py:235` |
| 5 | `industrial_device` | 工业设备干扰（主动发射机假设） | 900e6 | `eirp_equivalent_dbm`（已写死） | `calibration` 写明 `P_EIRP = P_cond + G_t - L_t` 与锚点说明（sources 占位） | 1.0 | `['indoor','ground','sidewalk']` | 6 | 0.30 / 0.20 / 0.50 | 同构 A2G；不是 PSD/EMI 口径 | 配置 `UAV_GA.py:175`；语义 `UAV_GA.py:235` |
| 6 | `satellite_ground` | Ku 地面端等效辐射源 | 12e9 | `eirp_equivalent_dbm`（已写死；并给 dBW 换算） | `calibration.eirp_range_dbm=(50,70)` 与 `eirp_range_dbw=(20,40)`；并给 `P_EIRP(dBW)=10log10(P_BUC)+G_t-L_t`（sources 占位） | 1.0 | `['ground','roof']` | 0.5 | 0.95 / 0.05 / 0.00 | 同构 A2G | 配置 `UAV_GA.py:186`；语义 `UAV_GA.py:235` |
| 7 | `cellular_ue_ul` | 蜂窝 UE 上行 | 2.6e9 |（最终功率由 FPC 计算） | FPC：`Pu=min(Pmax, P0+α·PL+10log10(M)+ΔTF+f(ΔTPC))`，其中 `PL≈FSPL(d_serv)+pl_extra`（代理） | 1.0 | `['street','sidewalk','indoor','air']` | 0（默认不启用） | 0.30 / 0.05 / 0.65 | 同构 A2G | 配置 `UAV_GA.py:136`；FPC `UAV_GA.py:1674`；标定 `UAV_GA.py:1738` |

## 2) 通用“干扰到达功率与聚合”公式（所有类型共用）

- 干扰路径损耗：`_interference_path_loss_db()`（`UAV_GA.py:2096`）
  - `PL = P_LoS·(FSPL+η_LoS) + (1-P_LoS)·(FSPL+η_NLoS) + 10log10(path_loss_factor·height_factor) + [P.1238] + [P.2109]`
  - FSPL：`calculate_fspl_db_at_freq()`（`UAV_GA.py:1999`）  
    `FSPL(dB)=20log10(d)+20log10(f)+20log10(4π/c)`
  - 室内附加：
    - ITU‑R P.1238：`_itu_p1238_indoor_loss_db()`（`UAV_GA.py:2071`）
    - ITU‑R P.2109（BEL）：`_itu_p2109_bel_loss_db()`（`UAV_GA.py:2080`）

- 到达功率：`effective_power_dbm = source.power - path_loss_db`（`UAV_GA.py:2271`）
  - 超覆盖半径附加衰减：`-20log10(d/coverage_radius)`（`UAV_GA.py:2271`）

- 频谱耦合：`_spectral_coupling_factor()`（`UAV_GA.py:870`）
  - 同频：`η=1`
  - `cochannel_only`：异频 `η=0`
  - `acir`：`η=10^(-ACIR/10)`

- 小尺度衰落（可选）：`_stable_exp_fading_gain()`（`UAV_GA.py:1791`），`h~Exp(1)`；关闭则 `h=1`

- 线性域聚合干扰功率：`calculate_interference_power_mw()`（`UAV_GA.py:2271`）
  - `I_y[mW] = Σ mw(P_rx_interf) · η(Δf) · h · activity`

