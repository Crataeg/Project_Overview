# 通信劣化评估公式（通俗报告）

这份报告用“普通人能理解”的方式说明：**通信劣化评估公式是什么、代码如何计算、每一项代表什么**。  
对应代码在 `UAV_GA.py` 的核心函数：
- `calculate_link_degradation(...)`
- `calculate_comm_degradation(...)`
- `calculate_speed_energy_efficiency_degradation(...)`
- `calculate_power_margin_components(...)`

---

## 1) 总体公式（最终的“综合劣化”）

每一条无人机–地面站链路的综合劣化度（范围 0~1，越大越差）：

```
D_total = ( w1 * D_comm + w2 * D_speed ) / (w1 + w2)
```

对应代码：`calculate_link_degradation(...)`  
默认权重：`w1=0.7`（通信劣化），`w2=0.3`（速度/能效劣化）

**直观理解**：  
“通信质量差 + 飞得太费电” 会共同拉高劣化值。  
权重大的一项影响更大。

---

## 2) 通信劣化 D_comm（默认使用“功率裕量”模型）

默认模型是 `margin`（见 `degradation_model` 默认值）。  
核心公式是一个“S 型映射”，把“功率裕量 M”转换成 0~1 的劣化度：

```
M = P_rx - P_IN
D_comm = 1 / (1 + exp((M - θ) / s))
```

对应代码：`calculate_comm_degradation(...)`  
其中：
- `P_rx`：接收信号功率（dBm）
- `P_IN`：干扰 + 噪声功率（dBm）
- `θ`：阈值（`margin_threshold_db`，默认 0）
- `s`：斜率（`margin_slope_db`，默认 3）

**直观理解**：  
功率裕量 M 越小（信号不够强、干扰噪声越大），劣化越高；  
当 M 低于阈值时，劣化快速逼近 1（很差）。

---

## 3) M 的来源：接收功率与干扰噪声

功率裕量：

```
M = P_rx - P_IN
```

其中：
```
P_IN = 10log10( I_total + N )
```

- `I_total`：所有干扰源功率之和（线性域 mW）
- `N`：噪声功率（mW）

对应代码：`calculate_power_margin_components(...)`  
会基于距离、路径损耗、遮挡等计算 `P_rx`，并汇总干扰源与噪声得到 `P_IN`。

**直观理解**：  
信号越强、干扰噪声越小 → M 越大 → 劣化越小。

---

## 4) 速度/能效劣化 D_speed（无人机飞得越费电越差）

速度劣化是基于旋翼无人机功率模型：

```
P(V) = P0(1+3V^2/U_tip^2) 
     + Pi*sqrt( sqrt(1+V^4/(4v0^4)) - V^2/(2v0^2) )
     + 0.5*d0*rho*s*A*V^3

D_speed = 1 - P_hover / P(V)
```

对应代码：`calculate_speed_energy_efficiency_degradation(...)`

**直观理解**：  
速度越快，推进功率 P(V) 越高，能效越差，劣化越大；  
悬停功率 `P_hover` 是“基准”，用来衡量相对劣化。

---

## 5) “综合劣化”怎么在报告里统计

在 `analyze_scenario_metrics(...)` 中会对所有无人机-地面站链路计算：

```
avg_total_deg = mean( D_total over all links )
```

这也是你在对比图里看到的“综合通信劣化”指标。  
**数值越大，说明整体通信场景越差**。

---

## 6) 一句话总结

**你的劣化公式可以理解为：**
> “通信信号比干扰噪声弱 + 无人机飞得很费电”  
> 会让整体劣化分数更高。

默认最终使用的综合公式就是：

```
D_total = (0.7 * D_comm + 0.3 * D_speed) / 1.0
```

---

如果你希望我把报告改成“数学推导版”或“论文风格版”，告诉我。  
如果你希望只保留 `margin` 这一路的公式并把其它分支隐藏，也可以告诉我。
