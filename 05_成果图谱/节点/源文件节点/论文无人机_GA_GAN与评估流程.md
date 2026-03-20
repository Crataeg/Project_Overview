---
title: "论文无人机 GA GAN 与评估流程"
---
# 论文无人机 GA GAN 与评估流程

## 节点总结

该节点对应无人机模型的实验流程层。`gan_uav_pipeline.py` 用 GA 样本训练 GAN 并生成场景，`compare_random_ga_gan.py` 对比 GA/GAN/Random，`evaluate.py` 负责 KPI、BLER 与吞吐评估。

## 原始文件位置

- `D:\UAV_Communication_GA\12\gan_uav_pipeline.py`
- `D:\UAV_Communication_GA\12\compare_random_ga_gan.py`
- `D:\UAV_Communication_GA\12\evaluate.py`
- `D:\UAV_Communication_GA\12\PROJECT_GUIDE.md`

## 关键内容

- `gan_uav_pipeline.py` 先用 `geatpy` 采样最优或最差场景，再把样本缩放后交给 PyTorch MLP GAN。
- `compare_random_ga_gan.py` 把 `GA / GAN / Random` 放到同一评估框架里做统计与可视化对比。
- `evaluate.py` 进一步把链路记录映射成 outage、throughput、EE、BLER_A、BLER_B 指标。
