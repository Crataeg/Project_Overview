# 对话沉淀

这里用于保存 Codex 对话结论、实验摘要、文献摘记和阶段性判断。

示例：

```powershell
@'
这里写入本次对话的中文总结。
'@ | python tools/obsidian/capture_knowledge_note.py --title "阶段总结" --tags 对话沉淀,知识更新 --stdin
```
