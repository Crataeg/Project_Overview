# Project Overview

这是一个用于 `Obsidian` 的工程总览知识库仓库，核心目标是把现有的工程结构、知识卡片、主题图谱和成果图谱统一纳入 Git 管理，便于持续维护、版本回溯和 GitHub 备份。

## 仓库结构

- `00_知识库总览.md`：知识库入口页
- `05_成果图谱/`：按成果组织的总览与节点关系
- `10_总索引/`：五大知识项的聚合索引
- `15_主题图谱/`：主题级笔记与专题材料
- `20_知识卡片/`：拆分后的细粒度知识卡片
- `30_对话沉淀/`：阶段性对话与结论沉淀
- `90_自动索引/`：自动生成的索引与元数据
- `.obsidian/`：当前 vault 的 Obsidian 配置与插件配置

## 管理约定

- Git 仓库根目录就是当前 vault 根目录，便于 `Obsidian Git` 直接识别
- 提交以知识库内容变更为主，`workspace.json` 等机器本地状态不纳入版本控制
- 默认分支建议使用 `main`
- 日常同步流程建议为：`Pull -> 编辑 -> Commit -> Push`

## 本地维护

- 重建知识库索引：`python tools\\obsidian\\build_obsidian_vault.py`
- 写入对话沉淀：`python tools\\obsidian\\capture_knowledge_note.py --title "中文标题" --stdin`
- Obsidian 中启用 `Obsidian Git` 后，可直接用命令面板执行提交、拉取和推送
