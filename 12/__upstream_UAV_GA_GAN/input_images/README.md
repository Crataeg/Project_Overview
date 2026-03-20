# input_images

把你在聊天里发的截图/图片保存为文件后，放到这个目录里（推荐按顺序命名：`01.png`、`02.png`…）。

然后在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe .\make_images_docx.py input_images outline_images.docx --title "Outline Images"
```

生成的 Word 文件为：`outline_images.docx`。

