---
title: "论文卫星 MATLAB 工程主程序"
---
# 论文卫星 MATLAB 工程主程序

## 节点总结

该节点对应 MATLAB 卫星模型的主执行层。`LEO_StarNet_EMC_V7_0_Engineering.m` 是 V7 工程版主程序，负责星座场景、上下行、干扰机、最坏工况搜索、图形与输出；`run_LEO_EMC_Sim.m` 是 Simulink 侧运行入口；`satellite.m` 则是更基础的通信链路仿真脚本。

## 原始文件位置

- `D:\一汽项目\LEO_Sim\LEO_Sim_V7_modified\v7proj\LEO_StarNet_EMC_V7_0_Engineering.m`
- `D:\一汽项目\LEO_Sim\run_LEO_EMC_Sim.m`
- `D:\一汽项目\satellite.m`

## 关键内容

- 主程序中明确写到是 “Engineering Delivery Version”，并包含二维星座视图、Sky View、频率复用、ISL 图、上下行 EMC 分析。
- `run_LEO_EMC_Sim.m` 用于运行 Simulink 模型并输出 `errRate=[BER,numErr,numBits]`。
- `satellite.m` 体现了更基础的通信仿真底座，包括 QPSK、AWGN、Turbo 编码和 BER 曲线。
