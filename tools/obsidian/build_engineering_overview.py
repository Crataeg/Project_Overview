from __future__ import annotations

import json
import os
import re
import shutil
import stat
import textwrap
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from xml.etree import ElementTree as ET

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None


ROOT = Path("D:/")
PROJECT_ROOTS = {
    "论文无人机": ROOT / "论文无人机" / "UAV_GA_GAN",
    "论文卫星": ROOT / "论文卫星" / "LEO_Sim",
    "专利一汽": ROOT / "专利一汽" / "LEO_EMCSim_Lab",
    "专利自有": ROOT / "专利自有" / "LEO_Sim_Patent",
    "正向设计规范": ROOT / "正向设计规范" / "Forward_Design",
}
TOTAL_VAULT = Path(r"D:\工程总览\总的obsidian知识库")

TEXT_EXTS = {
    ".c",
    ".cpp",
    ".csv",
    ".doc",
    ".json",
    ".m",
    ".md",
    ".mlx",
    ".py",
    ".svg",
    ".tex",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk")


@dataclass(frozen=True)
class CopyTask:
    label: str
    src: str
    dest_rel: str
    task_type: str
    summary: str
    writing_value: str
    entries: Sequence[str] = ()
    ignore_globs: Sequence[str] = ()


@dataclass(frozen=True)
class ProjectSpec:
    name: str
    description: str
    writing_summary: str
    cross_summary: str
    tasks: Sequence[CopyTask]


@dataclass
class TaskResult:
    label: str
    src: str
    dest: str
    task_type: str
    summary: str
    writing_value: str
    source_entries: List[str]
    file_count: int
    top_files: List[str]
    previews: Dict[str, str]


def sanitize_name(text: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", text.strip())
    value = re.sub(r"\s+", "_", value)
    return value[:80] or "未命名"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel_link(from_path: Path, to_path: Path) -> str:
    return os.path.relpath(to_path, from_path.parent).replace("\\", "/")


def project_root_for(project_name: str) -> Path:
    return PROJECT_ROOTS.get(project_name, ROOT / project_name)


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


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        force_remove(path)
    path.mkdir(parents=True, exist_ok=True)


def make_ignore(globs: Sequence[str]):
    def _ignore(_dir: str, names: List[str]) -> set[str]:
        skipped = set()
        for name in names:
            for pattern in globs:
                if Path(name).match(pattern):
                    skipped.add(name)
                    break
        return skipped

    return _ignore


def iter_files(base: Path) -> Iterable[Path]:
    if base.is_file():
        if not base.name.startswith("~$"):
            yield base
        return
    for path in sorted(base.rglob("*")):
        if path.is_file() and not path.name.startswith("~$"):
            yield path


def read_text(path: Path, max_chars: int = 1800) -> str:
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)[:max_chars]
        except Exception:
            continue
    return ""


def extract_docx_preview(path: Path, max_paragraphs: int = 18) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            data = zf.read("word/document.xml")
    except Exception:
        return ""
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return ""
    texts: List[str] = []
    for node in root.iter():
        if node.tag.endswith("}t") and node.text:
            value = node.text.strip()
            if value:
                texts.append(value)
    return "\n".join(texts[:max_paragraphs])


def extract_pdf_preview(path: Path, max_chars: int = 1500) -> str:
    if PdfReader is None:
        return "当前环境未安装 pypdf，未提取 PDF 文本预览。"
    try:
        reader = PdfReader(str(path))
        first_page = reader.pages[0].extract_text() if reader.pages else ""
        meta = reader.metadata or {}
        title = meta.get("/Title", "") if isinstance(meta, dict) else ""
        return "\n".join(x for x in [title, (first_page or "")[:max_chars].strip()] if x)
    except Exception as exc:
        return f"PDF 预览提取失败: {exc}"


def preview_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        return read_text(path)
    if ext == ".docx":
        return extract_docx_preview(path)
    if ext == ".pdf":
        return extract_pdf_preview(path)
    if ext == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                return "\n".join(zf.namelist()[:20])
        except Exception as exc:
            return f"压缩包预览失败: {exc}"
    return ""


def pick_top_files(base: Path, limit: int = 12) -> List[Path]:
    files = list(iter_files(base))
    priority_exts = {
        ".py": 0,
        ".m": 0,
        ".docx": 1,
        ".pdf": 1,
        ".md": 2,
        ".txt": 2,
        ".json": 3,
        ".svg": 4,
        ".png": 5,
        ".zip": 6,
    }

    def sort_key(path: Path):
        return (priority_exts.get(path.suffix.lower(), 9), len(path.parts), path.name.lower())

    return sorted(files, key=sort_key)[:limit]


def copy_task(project_root: Path, task: CopyTask) -> TaskResult:
    src = Path(task.src)
    dest = project_root / task.dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    ignore = make_ignore(task.ignore_globs)

    if task.entries:
        ensure_clean_dir(dest)
        for name in task.entries:
            item = src / name
            if not item.exists():
                continue
            target = dest / name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True, ignore=ignore)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
        file_base = dest
    elif src.is_dir():
        ensure_clean_dir(dest)
        shutil.copytree(src, dest, dirs_exist_ok=True, ignore=ignore)
        file_base = dest
    else:
        if dest.exists():
            if dest.is_dir():
                force_remove(dest)
            else:
                force_remove(dest)
        shutil.copy2(src, dest)
        file_base = dest

    copied_files = list(iter_files(file_base))
    top_files = pick_top_files(file_base)
    previews = {}
    for path in top_files[:3]:
        text = preview_text(path)
        if text:
            previews[str(path.relative_to(project_root))] = text[:1400]

    return TaskResult(
        label=task.label,
        src=str(src),
        dest=str(dest),
        task_type=task.task_type,
        summary=task.summary,
        writing_value=task.writing_value,
        source_entries=list(task.entries),
        file_count=len(copied_files),
        top_files=[str(path.relative_to(project_root)) for path in top_files],
        previews=previews,
    )


def write_obsidian_config(vault: Path) -> None:
    cfg_dir = vault / ".obsidian"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        cfg_dir / "app.json",
        json.dumps({"alwaysUpdateLinks": True, "promptDelete": False}, ensure_ascii=False, indent=2),
    )
    write_text(
        cfg_dir / "core-plugins.json",
        json.dumps(["file-explorer", "search", "backlink", "graph", "outgoing-link"], ensure_ascii=False, indent=2),
    )


def project_note_paths(vault: Path, task: TaskResult) -> Path:
    folder = "10_成果节点" if task.task_type == "成果" else "20_参考节点"
    return vault / folder / f"{sanitize_name(task.label)}.md"


def build_task_note(vault: Path, project_root: Path, task: TaskResult) -> None:
    current = project_note_paths(vault, task)
    lines = [
        "---",
        f'title: "{task.label}"',
        f'task_type: "{task.task_type}"',
        f'source_path: "{task.src}"',
        f'copied_path: "{task.dest}"',
        "---",
        f"# {task.label}",
        "",
        "## 节点总结",
        "",
        task.summary,
        "",
        "## 写作价值",
        "",
        task.writing_value,
        "",
        "## 原始位置",
        "",
        f"- `{task.src}`",
        "",
    ]
    if task.source_entries:
        lines.extend(
            [
                "## 来源条目",
                "",
            ]
        )
        for name in task.source_entries:
            lines.append(f"- `{name}`")
        lines.extend([""])
    lines.extend(
        [
            "## 复制后位置",
            "",
            f"- `{task.dest}`",
            "",
            f"## 文件规模",
            "",
            f"- 文件数：`{task.file_count}`",
            "",
            "## 关键文件",
            "",
        ]
    )
    for file_rel in task.top_files:
        lines.append(f"- `{file_rel}`")
    if task.previews:
        lines.extend(["", "## 代表性内容预览", ""])
        for file_rel, preview in task.previews.items():
            lines.extend([f"### `{file_rel}`", "", "```text", preview, "```", ""])
    write_text(current, "\n".join(lines))


def build_project_vault(project_root: Path, spec: ProjectSpec, results: Sequence[TaskResult]) -> None:
    vault = project_root / "obsidian知识库"
    ensure_clean_dir(vault)
    write_obsidian_config(vault)

    result_notes = [project_note_paths(vault, x) for x in results if x.task_type == "成果"]
    ref_notes = [project_note_paths(vault, x) for x in results if x.task_type == "参考"]

    for item in results:
        build_task_note(vault, project_root, item)

    overview = vault / "00_项目总览.md"
    lines = [
        f"# {spec.name}",
        "",
        spec.description,
        "",
        "## 项目结构",
        "",
        f"- 成果本身：`{project_root / '成果本身'}`",
        f"- 参考文献：`{project_root / '参考文献'}`",
        f"- Obsidian 知识库：`{vault}`",
        f"- 更新文件夹：`{project_root / '更新文件夹'}`",
        "",
        "## 成果节点",
        "",
    ]
    for note in result_notes:
        lines.append(f"- [{note.stem}]({rel_link(overview, note)})")
    lines.extend(["", "## 参考节点", ""])
    for note in ref_notes:
        lines.append(f"- [{note.stem}]({rel_link(overview, note)})")
    lines.extend(
        [
            "",
            "## 技术涉及文献、报告与写作基础",
            "",
            spec.writing_summary,
            "",
            "## 跨项目关联",
            "",
            spec.cross_summary,
            "",
        ]
    )
    write_text(overview, "\n".join(lines))

    synthesis = vault / "30_写作基础" / "技术涉及文献与报告总结.md"
    blocks = [
        f"# {spec.name} 技术涉及文献与报告总结",
        "",
        "## 总结",
        "",
        spec.writing_summary,
        "",
        "## 参考材料作用分解",
        "",
    ]
    for item in [x for x in results if x.task_type == "参考"]:
        note = project_note_paths(vault, item)
        blocks.append(f"- [{item.label}]({rel_link(synthesis, note)})：{item.writing_value}")
    write_text(synthesis, "\n".join(blocks))


def write_update_log(project_root: Path, spec: ProjectSpec, results: Sequence[TaskResult]) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = project_root / "更新文件夹" / f"{stamp}_工程迁移与知识库初始化.md"
    lines = [
        f"# {spec.name} 工程迁移与知识库初始化",
        "",
        f"- 更新时间：`{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- 项目目录：`{project_root}`",
        "",
        "## 本次动作",
        "",
        "- 按“成果本身 / 参考文献 / obsidian知识库 / 更新文件夹”重建项目结构。",
        "- 将原始工程、文献、报告复制到当前项目目录下的对应子文件夹。",
        "- 生成项目 Obsidian 子知识库并补充技术总结、文献作用和写作基础说明。",
        "",
        "## 复制映射",
        "",
    ]
    for item in results:
        lines.append(f"- `{item.src}` -> `{item.dest}`")
    write_text(path, "\n".join(lines))


def build_total_vault(projects: Sequence[ProjectSpec]) -> None:
    ensure_clean_dir(TOTAL_VAULT)
    write_obsidian_config(TOTAL_VAULT)

    def task_note(project_name: str, task_type: str, label: str) -> Path:
        folder = "10_成果节点" if task_type == "成果" else "20_参考节点"
        return project_root_for(project_name) / "obsidian知识库" / folder / f"{sanitize_name(label)}.md"

    overview = TOTAL_VAULT / "00_工程总览.md"
    fusion_nav = TOTAL_VAULT / "20_交叉图谱" / "新增资产融合导航.md"
    lines = [
        "# 工程总览",
        "",
        "这个总仓库用于统筹五条工作线，并把五个子 Obsidian 知识库连接起来。",
        "",
        "## 五条工作线",
        "",
    ]
    for spec in projects:
        project_root = project_root_for(spec.name)
        project_note = project_root / "obsidian知识库" / "00_项目总览.md"
        lines.append(f"- [{spec.name}]({rel_link(overview, project_note)})：{spec.description}")
    lines.extend(
        [
            "",
            "## 共用观察",
            "",
            "- `专利自有 / 专利一汽 / 论文卫星 / 正向设计规范` 四条线共享低轨卫星通信、EMC、标准规范、专利对标和技术报告。",
            "- `论文无人机` 独立成线，但在“通信性能评估、最坏工况搜索、链路指标建模”上可与卫星线形成方法参考关系。",
            "- 总仓库建议作为跨项目导航与主题沉淀入口，各项目子仓库负责本项目内部的详细拆解。",
            "",
            "## 新增资产融合",
            "",
            f"- [新增资产融合导航]({rel_link(overview, fusion_nav)})：汇总 `D:\\无人机通信系统抗干扰性能的测试评估技术研究` 与 `D:\\一汽项目` 的分类落位。",
            "",
        ]
    )
    write_text(overview, "\n".join(lines))

    fusion_lines = [
        "# 新增资产融合导航",
        "",
        "## 总体说明",
        "",
        "- 本次把 `D:\\无人机通信系统抗干扰性能的测试评估技术研究` 和 `D:\\一汽项目` 的资产按“成果 / 参考 / 写作 / 代码底座 / 标定建设 / 专利素材”重新分类并复制到五个项目库中。",
        "- 总库不重复镜像整套资产，而是通过这里的分类导航指向各子项目中的实际复制落位。",
        "",
        "## 无人机测试评估技术研究 -> 子项目",
        "",
        f"- 论文无人机 / 代码与设计：[{sanitize_name('无人机研究_代码工程与设计文档')}]({rel_link(fusion_nav, task_note('论文无人机', '成果', '无人机研究_代码工程与设计文档'))})",
        f"- 论文无人机 / 论文报告与汇报：[{sanitize_name('无人机研究_论文报告与汇报材料')}]({rel_link(fusion_nav, task_note('论文无人机', '成果', '无人机研究_论文报告与汇报材料'))})",
        f"- 论文无人机 / 专题论文与参考：[{sanitize_name('无人机研究_专题论文与参考资料')}]({rel_link(fusion_nav, task_note('论文无人机', '参考', '无人机研究_专题论文与参考资料'))})",
        f"- 论文无人机 / 标定与实验建设：[{sanitize_name('无人机研究_测试标定与实验建设资料')}]({rel_link(fusion_nav, task_note('论文无人机', '参考', '无人机研究_测试标定与实验建设资料'))})",
        f"- 正向设计规范 / 测试评估方法迁移：[{sanitize_name('无人机研究_测试评估方法与实验室建设资料')}]({rel_link(fusion_nav, task_note('正向设计规范', '参考', '无人机研究_测试评估方法与实验室建设资料'))})",
        "",
        "## 一汽项目 -> 子项目",
        "",
        f"- 论文卫星 / 主工程：[{sanitize_name('MATLAB卫星仿真工程')}]({rel_link(fusion_nav, task_note('论文卫星', '成果', 'MATLAB卫星仿真工程'))})",
        f"- 论文卫星 / 原型工程与 Simulink：[{sanitize_name('卫星EMC原型工程与Simulink模型')}]({rel_link(fusion_nav, task_note('论文卫星', '成果', '卫星EMC原型工程与Simulink模型'))})",
        f"- 论文卫星 / 干扰识别数据与训练模型：[{sanitize_name('卫星干扰识别数据与训练模型')}]({rel_link(fusion_nav, task_note('论文卫星', '成果', '卫星干扰识别数据与训练模型'))})",
        f"- 论文卫星 / 研究方案与补充报告：[{sanitize_name('一汽项目_技术报告与简版方案')}]({rel_link(fusion_nav, task_note('论文卫星', '参考', '一汽项目_技术报告与简版方案'))})",
        f"- 专利自有 / 工程与数据底座：[{sanitize_name('一汽项目_卫星EMC原型工程与数据集')}]({rel_link(fusion_nav, task_note('专利自有', '成果', '一汽项目_卫星EMC原型工程与数据集'))})",
        f"- 专利自有 / 专利对标：[{sanitize_name('主机厂专利与EMC对标资料')}]({rel_link(fusion_nav, task_note('专利自有', '参考', '主机厂专利与EMC对标资料'))})",
        f"- 专利一汽 / 专利图示与模板：[{sanitize_name('一汽项目_专利图示素材与模板')}]({rel_link(fusion_nav, task_note('专利一汽', '成果', '一汽项目_专利图示素材与模板'))})",
        f"- 专利一汽 / 试验室仿真支撑简版：[{sanitize_name('一汽项目_试验室仿真支撑简版资料')}]({rel_link(fusion_nav, task_note('专利一汽', '成果', '一汽项目_试验室仿真支撑简版资料'))})",
        f"- 专利一汽 / 专利与标准参考：[{sanitize_name('一汽项目_专利与标准参考合集')}]({rel_link(fusion_nav, task_note('专利一汽', '参考', '一汽项目_专利与标准参考合集'))})",
        f"- 正向设计规范 / 研究方案根目录定稿：[{sanitize_name('一汽项目_根目录方案简版与定稿')}]({rel_link(fusion_nav, task_note('正向设计规范', '成果', '一汽项目_根目录方案简版与定稿'))})",
        f"- 正向设计规范 / 项目总结与学位论文：[{sanitize_name('一汽项目_项目总结与学位论文材料')}]({rel_link(fusion_nav, task_note('正向设计规范', '参考', '一汽项目_项目总结与学位论文材料'))})",
        f"- 正向设计规范 / 标准与对标合集：[{sanitize_name('一汽项目_标准与对标参考合集')}]({rel_link(fusion_nav, task_note('正向设计规范', '参考', '一汽项目_标准与对标参考合集'))})",
        "",
        "## 使用建议",
        "",
        "- 先从总库导航进入分类节点，再从节点跳转到实际复制目录查看资产本体。",
        "- 论文写作优先看 `论文无人机 / 论文卫星`，规范与方案优先看 `正向设计规范`，专利材料优先看 `专利一汽 / 专利自有`。",
    ]
    write_text(fusion_nav, "\n".join(fusion_lines))

    satellite_cross = TOTAL_VAULT / "20_交叉图谱" / "卫星仿真与专利主线.md"
    lines = [
        "# 卫星仿真与专利主线",
        "",
        "## 主线总结",
        "",
        "卫星相关四条线不是孤立存在的：MATLAB 工程模型负责形成仿真底座，专利线负责把技术方案整理成专利文本与附图，正向设计规范线负责把工程目标与标准要求固定下来，论文卫星线负责把仿真与文献支撑转化为论文表达。",
        "",
        "## 关联项目",
        "",
    ]
    for name in ["专利自有", "专利一汽", "论文卫星", "正向设计规范"]:
        note = project_root_for(name) / "obsidian知识库" / "00_项目总览.md"
        lines.append(f"- [{name}]({rel_link(satellite_cross, note)})")
    write_text(satellite_cross, "\n".join(lines))

    uav_cross = TOTAL_VAULT / "20_交叉图谱" / "无人机仿真主线.md"
    write_text(
        uav_cross,
        "\n".join(
            [
                "# 无人机仿真主线",
                "",
                "## 主线总结",
                "",
                "无人机线以 Python 仿真工程、GA+GAN 场景生成、KPI/BLER 评估和公开实测对照为核心，适合作为论文写作与方法迁移的独立研究线。",
                "",
                "## 关联项目",
                "",
                f"- [论文无人机]({rel_link(uav_cross, project_root_for('论文无人机') / 'obsidian知识库' / '00_项目总览.md')})",
                f"- [卫星仿真与专利主线]({rel_link(uav_cross, satellite_cross)})",
            ]
        ),
    )


def build_project_structure(project_root: Path) -> None:
    (project_root / "成果本身").mkdir(parents=True, exist_ok=True)
    (project_root / "参考文献").mkdir(parents=True, exist_ok=True)
    (project_root / "obsidian知识库").mkdir(parents=True, exist_ok=True)
    (project_root / "更新文件夹").mkdir(parents=True, exist_ok=True)


PROJECTS: Sequence[ProjectSpec] = [
    ProjectSpec(
        name="专利自有",
        description="放置 MATLAB 卫星仿真专利项目，重点保留仿真工程、技术报告、专利对标与支撑文献。",
        writing_summary=(
            "这一条线的写作基础由三部分组成：第一，`LEO_Sim` 与 `satellite.m` 给出可落地的 MATLAB 仿真底座；"
            "第二，`LEO_EMC_Sim`、数据集与训练模型补齐了原型级 Simulink 链路和干扰识别训练材料；"
            "第三，试验室环境下低轨卫星通信系统 EMC 仿真技术相关报告把工程问题和方法路线固定下来；"
            "第四，专利与车载卫星通信资料提供现有方案对标。后续写作时可以直接围绕“仿真场景、链路建模、EMC 指标、最坏工况、专利差异化设计点”展开。"
        ),
        cross_summary="它与 `论文卫星` 共享 MATLAB 仿真底座，与 `专利一汽` 共享卫星通信专利表达方式，与 `正向设计规范` 共享标准和研究方案。",
        tasks=[
            CopyTask(
                label="MATLAB卫星仿真工程",
                src=r"D:\一汽项目\LEO_Sim",
                dest_rel=r"成果本身\代码工程\LEO_Sim",
                task_type="成果",
                summary="该节点保留 MATLAB 低轨卫星通信 EMC 仿真工程本体，是专利自有线的核心技术成果。",
                writing_value="可直接支撑专利技术效果、系统结构、参数配置、最坏工况搜索与结果论证。",
            ),
            CopyTask(
                label="基础卫星链路脚本",
                src=r"D:\一汽项目\satellite.m",
                dest_rel=r"成果本身\代码工程\satellite.m",
                task_type="成果",
                summary="该脚本提供更基础的卫星链路通信仿真入口，可作为专利技术方案的底层算法说明。",
                writing_value="适合写成“基础通信链路模型”或“仿真验证底座”章节。",
            ),
            CopyTask(
                label="试验室卫星EMC仿真技术报告",
                src=r"D:\一汽项目\试验室环境下低轨卫星通信系统EMC仿真技术",
                dest_rel=r"成果本身\报告材料\试验室环境下低轨卫星通信系统EMC仿真技术",
                task_type="成果",
                summary="该目录包含专利自有线所需的技术报告、标准和论文支撑，是把仿真工程转成专利方案的重要报告层材料。",
                writing_value="有助于提炼“为何这样建模、为何这样定义测试链路、为何这样设计指标”的论证语言。",
            ),
            CopyTask(
                label="一汽项目_卫星EMC原型工程与数据集",
                src=r"D:\一汽项目",
                dest_rel=r"成果本身\代码工程\一汽项目_卫星EMC原型工程与数据集",
                task_type="成果",
                summary="该节点汇总一汽项目中的 Simulink 原型工程、根目录系统模型、STFT 数据集和训练模型，是专利自有线继承更早期工程实现的重要补充。",
                writing_value="可支撑专利中对原型验证链、识别模块、数据集构建与训练依据的补充说明。",
                entries=(
                    "LEO_EMC_Sim",
                    "dataset_stft_r2021a",
                    "GAN_Jammer_R2021a.mat",
                    "lenet_stft_model_r2021a.mat",
                    "build_LEO_EMC_Sim.m",
                    "LEO_EMC_System.slx",
                    "LEO_EMC_System.slx.autosave",
                    "LEO_EMC_System.slxc",
                    "slprj",
                ),
            ),
            CopyTask(
                label="卫星通信专利对标",
                src=r"D:\一汽项目\车载卫星通信系统",
                dest_rel=r"参考文献\专利对标\车载卫星通信系统",
                task_type="参考",
                summary="该目录提供车载卫星通信相关现有专利方案，可用于界定自有专利的新颖点和差异点。",
                writing_value="适合在撰写背景技术、现有方案不足与创新点时直接引用。",
            ),
            CopyTask(
                label="主机厂专利与EMC对标资料",
                src=r"D:\一汽项目\专利",
                dest_rel=r"参考文献\专利对标\主机厂专利与EMC资料",
                task_type="参考",
                summary="该目录整理了一汽、吉利、比亚迪、赛力斯、长安等主机厂的专利和 EMC 相关资料。",
                writing_value="可以支持竞品对标、已有技术路线分析以及专利布局比较。",
            ),
        ],
    ),
    ProjectSpec(
        name="专利一汽",
        description="放置 CST 卫星通信仿真专利项目，重点保留专利文本、附图、教程以及支撑论文标准。",
        writing_summary=(
            "这一条线已经具备交底书、附图清单、附图压缩包和 CST 教程，写作基础相对完整。"
            "后续只需要继续强化三部分：一是从交底书中稳定技术方案口径，二是从附图流程图中梳理方法步骤，三是用标准与论文把每一个判定环节和参数换算环节补足依据。"
        ),
        cross_summary="它与 `专利自有` 共用卫星通信对标资料，与 `论文卫星` 共用卫星仿真结果和文献基础，与 `正向设计规范` 共用标准规范与研究方案。",
        tasks=[
            CopyTask(
                label="一汽专利文本与附图",
                src=r"D:\专利一汽\LEO_EMCSim_Lab\更新文件夹\20260323_GitHub上传下载归档\from_Project_Overview根目录\tmp_leo_emcsim_lab_remote",
                dest_rel=r"成果本身\专利文本与附图\专利一汽",
                task_type="成果",
                summary="该目录包含一汽专利项目的交底书、附图清单、SVG/PNG 附图和压缩包，是当前最直接的专利成果载体。",
                writing_value="可直接形成专利说明书正文、附图说明、流程说明和实施方式。",
            ),
            CopyTask(
                label="CST卫星EMC支撑报告",
                src=r"D:\一汽项目\试验室环境下低轨卫星通信系统EMC仿真技术",
                dest_rel=r"成果本身\支撑报告\试验室环境下低轨卫星通信系统EMC仿真技术",
                task_type="成果",
                summary="该目录提供专利一汽线的技术报告、标准和论文支撑，是从仿真思路过渡到专利论证的报告层材料。",
                writing_value="可用于解释交底书中的测试环境、场强换算、判定逻辑和输出页面设计依据。",
            ),
            CopyTask(
                label="一汽项目_试验室仿真支撑简版资料",
                src=r"D:\一汽项目",
                dest_rel=r"成果本身\支撑报告\一汽项目补充试验室仿真材料",
                task_type="成果",
                summary="该节点收集一汽项目根目录中的试验室仿真技术定稿、一页纸与两页纸版本，便于专利线在交付口径、简版汇报和专利支撑材料之间切换。",
                writing_value="适合在专利申报、项目汇报和对外说明时快速抽取不同粒度的技术表述。",
                entries=(
                    "试验室环境下低轨卫星通信系统EMC仿真技术.docx",
                    "试验室环境下低轨卫星通信系统EMC仿真技术（一页纸简化版）.docx",
                    "试验室环境下低轨卫星通信系统EMC仿真技术（两页纸简化版）.docx",
                    "（一页纸）试验室环境下低轨卫星通信系统EMC仿真技术.docx",
                    "（一页纸）试验室环境下低轨卫星通信系统EMC仿真技术(1).docx",
                    "（两页纸）试验室环境下低轨卫星通信系统EMC仿真技术.docx",
                    "lab_emc_sim_one_page.docx",
                    "lab_emc_sim_two_page.docx",
                    "最终稿_格式对齐_试验室环境下低轨卫星通信系统EMC仿真技术.docx",
                ),
            ),
            CopyTask(
                label="一汽项目_专利图示素材与模板",
                src=r"D:\一汽项目",
                dest_rel=r"成果本身\专利文本与附图\一汽项目补充素材",
                task_type="成果",
                summary="该节点保留一汽项目根目录中的图片、模板和临时文档，可作为专利附图、版式与补充素材归档位。",
                writing_value="有利于后续整理附图、版式样例、专利说明书封面或汇报图示素材。",
                entries=(
                    "69bb216e-37c8-4c63-a17d-92b18d2c789d.png",
                    "6bd2a610-225a-470a-ba27-3a1f3e3b86ad.jpg",
                    "微信图片_20260122141540_92_31.png",
                    "模板.docx",
                    "新建 DOCX 文档.docx",
                ),
            ),
            CopyTask(
                label="卫星通信专利对标",
                src=r"D:\一汽项目\车载卫星通信系统",
                dest_rel=r"参考文献\专利对标\车载卫星通信系统",
                task_type="参考",
                summary="该目录包含车载卫星通信系统及车辆相关专利，可用作一汽专利项目的对标资料。",
                writing_value="有助于提炼背景技术和创新点边界。",
            ),
            CopyTask(
                label="主机厂专利与EMC对标资料",
                src=r"D:\一汽项目\专利",
                dest_rel=r"参考文献\专利对标\主机厂专利与EMC资料",
                task_type="参考",
                summary="该目录提供多家主机厂相关专利与 EMC 材料，是专利布局对照的重要参考。",
                writing_value="可用于专利检索、对标与撰写中的现有技术比较。",
            ),
            CopyTask(
                label="一汽项目_专利与标准参考合集",
                src=r"D:\一汽项目",
                dest_rel=r"参考文献\专利对标\一汽项目专利与标准合集",
                task_type="参考",
                summary="该节点汇总一汽项目根目录中的标准、专利与车载卫星通信单篇资料，可作为专利一汽线的补充参考合集。",
                writing_value="适合用于补齐背景技术、标准依据与典型专利边界说明。",
                entries=("参考文献", "车载卫星通信系统"),
            ),
        ],
    ),
    ProjectSpec(
        name="论文卫星",
        description="放置 MATLAB 卫星仿真论文项目，重点保留仿真工程、论文报告、文献与标准。",
        writing_summary=(
            "这一条线的论文写作基础最完整：MATLAB 仿真工程负责方法与结果，`LEO_Sim` 内报告负责工程解释，"
            "`LEO_EMC_Sim`、STFT 数据集与训练模型补充了原型级验证和识别支链，正向设计模板、研究方案论文和试验室仿真技术论文负责理论支撑与文献综述。"
            "后续论文可按“研究背景、系统模型、仿真方法、结果分析、EMC 讨论”展开。"
        ),
        cross_summary="它是卫星相关几条线的论文表达主线，与 `专利自有` 共用工程底座，与 `专利一汽` 共用专利表达目标，与 `正向设计规范` 共用标准与方案依据。",
        tasks=[
            CopyTask(
                label="MATLAB卫星仿真工程",
                src=r"D:\一汽项目\LEO_Sim",
                dest_rel=r"成果本身\代码工程\LEO_Sim",
                task_type="成果",
                summary="该目录是论文卫星线的核心仿真工程，包含 V7 工程版主程序、配置、结果和说明报告。",
                writing_value="可以直接支撑论文的方法部分、实验设置部分和结果分析部分。",
            ),
            CopyTask(
                label="基础卫星链路脚本",
                src=r"D:\一汽项目\satellite.m",
                dest_rel=r"成果本身\代码工程\satellite.m",
                task_type="成果",
                summary="该脚本提供基础卫星链路仿真模型，是论文卫星线的底层通信模型支撑。",
                writing_value="适合写成系统模型或基础仿真验证的补充说明。",
            ),
            CopyTask(
                label="试验室卫星EMC仿真报告",
                src=r"D:\一汽项目\试验室环境下低轨卫星通信系统EMC仿真技术",
                dest_rel=r"成果本身\报告材料\试验室环境下低轨卫星通信系统EMC仿真技术",
                task_type="成果",
                summary="该目录补充了论文卫星线的报告材料、标准与专题论文。",
                writing_value="有助于把工程仿真语言转成论文语体，并为实验设计与结果解读提供上下文。",
            ),
            CopyTask(
                label="卫星EMC原型工程与Simulink模型",
                src=r"D:\一汽项目",
                dest_rel=r"成果本身\代码工程\LEO_EMC_Sim_原型与Simulink",
                task_type="成果",
                summary="该节点保留一汽项目中的 Simulink 原型工程、根目录系统模型和构建缓存，是论文卫星线分析平台演进过程的重要工程补充。",
                writing_value="适合在论文中说明平台从原型到工程化版本的演进路线与验证链。",
                entries=(
                    "LEO_EMC_Sim",
                    "build_LEO_EMC_Sim.m",
                    "LEO_EMC_System.slx",
                    "LEO_EMC_System.slx.autosave",
                    "LEO_EMC_System.slxc",
                    "slprj",
                ),
            ),
            CopyTask(
                label="卫星干扰识别数据与训练模型",
                src=r"D:\一汽项目",
                dest_rel=r"成果本身\代码工程\干扰识别数据与训练模型",
                task_type="成果",
                summary="该节点汇总 STFT 数据集、GAN 干扰模型和 LeNet 训练模型，是论文卫星线干扰识别支链的直接数据底座。",
                writing_value="可支撑论文中的数据集构建、训练流程、识别实验与模型复现实验说明。",
                entries=("dataset_stft_r2021a", "GAN_Jammer_R2021a.mat", "lenet_stft_model_r2021a.mat"),
            ),
            CopyTask(
                label="卫星论文与模板文献",
                src=r"D:\一汽项目\EMC正向设计写作模板参考",
                dest_rel=r"参考文献\论文文献\EMC正向设计写作模板参考",
                task_type="参考",
                summary="该目录收集了卫星、空间站和 EMC 正向设计相关论文模板，是论文卫星线的重要综述与写作参考来源。",
                writing_value="适合用于引言、相关工作和写作结构借鉴。",
            ),
            CopyTask(
                label="正向设计研究方案论文与标准",
                src=r"D:\一汽项目\车载低轨卫星通信系统EMC性能正向设计技术研究方案",
                dest_rel=r"参考文献\研究方案与标准\车载低轨卫星通信系统EMC性能正向设计技术研究方案",
                task_type="参考",
                summary="该目录包含研究方案正文、标准、对照文件和专题论文，是论文卫星线的项目背景与规范依据。",
                writing_value="可直接支撑研究背景、工程需求、评价指标与标准约束部分。",
            ),
            CopyTask(
                label="一汽项目_技术报告与简版方案",
                src=r"D:\一汽项目",
                dest_rel=r"参考文献\技术报告\一汽项目技术报告与简版方案",
                task_type="参考",
                summary="该节点收纳一汽项目根目录中的项目总结、学位论文和试验室仿真简版材料，可作为论文卫星线的补充技术报告库。",
                writing_value="适合提炼项目背景、研究现状、工程意义和简版汇报中的成熟表述。",
                entries=(
                    "00 技术开发项目总结报告.docx",
                    "技术开发项目总结报告-电动智能网联汽车电磁兼容仿真验证技术研究.docx",
                    "16111047-宋亚丽-博士学位论文终稿.doc",
                    "试验室环境下低轨卫星通信系统EMC仿真技术.docx",
                    "试验室环境下低轨卫星通信系统EMC仿真技术（一页纸简化版）.docx",
                    "试验室环境下低轨卫星通信系统EMC仿真技术（两页纸简化版）.docx",
                    "（一页纸）试验室环境下低轨卫星通信系统EMC仿真技术.docx",
                    "（一页纸）试验室环境下低轨卫星通信系统EMC仿真技术(1).docx",
                    "（两页纸）试验室环境下低轨卫星通信系统EMC仿真技术.docx",
                    "lab_emc_sim_one_page.docx",
                    "lab_emc_sim_two_page.docx",
                    "最终稿_格式对齐_试验室环境下低轨卫星通信系统EMC仿真技术.docx",
                ),
            ),
            CopyTask(
                label="一汽项目_参考文献与标准合集",
                src=r"D:\一汽项目",
                dest_rel=r"参考文献\研究方案与标准\一汽项目参考文献合集",
                task_type="参考",
                summary="该节点汇总一汽项目根目录中的专利、标准和卫星通信参考资料，可作为论文卫星线的补充背景文献库。",
                writing_value="适合用于补充标准引用、专利背景和相关工作对照。",
                entries=("参考文献",),
            ),
        ],
    ),
    ProjectSpec(
        name="论文无人机",
        description="放置无人机 Python 仿真论文项目，重点保留代码工程、论文草稿与参考文献。",
        writing_summary=(
            "这一条线已经形成较完整的论文基础：`12` 目录中有 GA+GAN 场景生成、KPI/BLER 评估、实测对照输出，`论文` 目录中有论文初稿，"
            "`key reference` 中有综述、链路分析、评估方法、仿真证据链；新增导入的“无人机通信系统抗干扰性能的测试评估技术研究”目录进一步补齐了 0315 工程、论文级报告、干扰源标定、实验室建设和组会材料。"
            "后续论文可围绕“场景建模、最差场景搜索、指标定义、测量对照验证”组织正文。"
        ),
        cross_summary="它在研究对象上独立，但在通信性能评估、最坏工况搜索与指标构造上，可以为卫星线提供方法参考。",
        tasks=[
            CopyTask(
                label="Python无人机仿真工程_12",
                src=r"D:\UAV_Communication_GA\12",
                dest_rel=r"成果本身\代码工程\12",
                task_type="成果",
                summary="该目录是无人机论文线的主工程，包含 GA、GAN、KPI/BLER 评估和实测对照输出。",
                writing_value="是论文方法与结果章节的直接来源。",
                ignore_globs=("*.zip", ".venv*", "__pycache__", ".idea", ".vscode"),
            ),
            CopyTask(
                label="Python无人机模块化代码",
                src=r"D:\UAV_Communication_GA\UAV_Communication_GA",
                dest_rel=r"成果本身\代码工程\UAV_Communication_GA",
                task_type="成果",
                summary="该目录保留无人机通信工程的模块化实现，可作为论文中的模型封装和子模块说明依据。",
                writing_value="适合支撑模块设计、实现结构和可复现实验描述。",
                ignore_globs=("__pycache__",),
            ),
            CopyTask(
                label="论文草稿与写作材料",
                src=r"D:\UAV_Communication_GA\论文",
                dest_rel=r"成果本身\论文草稿\论文",
                task_type="成果",
                summary="该目录包含论文初稿，是无人机论文线现有写作成果的直接载体。",
                writing_value="后续论文继续写作、改写和结构优化时可直接在此基础上推进。",
                ignore_globs=("~$*",),
            ),
            CopyTask(
                label="无人机参考文献",
                src=r"D:\UAV_Communication_GA\key reference",
                dest_rel=r"参考文献\论文文献\key reference",
                task_type="参考",
                summary="该目录系统整理了无人机通信、轨迹优化、链路评估、仿真证据链等文献，是论文无人机线的核心参考库。",
                writing_value="可直接为引言、相关工作、方法依据和实验对照提供文献支撑。",
            ),
            CopyTask(
                label="无人机研究_代码工程与设计文档",
                src=r"D:\无人机通信系统抗干扰性能的测试评估技术研究",
                dest_rel=r"成果本身\代码工程\无人机通信测试评估技术研究_代码与设计",
                task_type="成果",
                summary="该节点汇总无人机测试评估技术研究目录中的 0315 论文级工程、早期模块工程以及代码设计说明文档，是论文无人机线的重要补充成果。",
                writing_value="可直接支撑论文方法、代码实现、参数设计与城市模型生成逻辑说明。",
                entries=("0315", "UAV_Communication_GA", "代码", "代码.docx", "城市模型生成.docx", "数据标定.docx", "最终代码.docx"),
                ignore_globs=("*.pyc",),
            ),
            CopyTask(
                label="无人机研究_论文报告与汇报材料",
                src=r"D:\无人机通信系统抗干扰性能的测试评估技术研究",
                dest_rel=r"成果本身\论文草稿\无人机通信测试评估技术研究_报告与汇报",
                task_type="成果",
                summary="该节点收纳无人机测试评估技术研究目录中的论文级报告、大纲、组会材料和中间文档，可作为论文写作与阶段汇报的连续证据链。",
                writing_value="有利于快速复用已经形成的长报告、短报告和汇报 PPT 内容，保持写作和汇报口径一致。",
                entries=(
                    "组会",
                    "1223",
                    "1224",
                    "小论文前期.pptx",
                    "无人机通信劣化-论文大纲_公式渲染版.docx",
                    "无人机通信劣化-论文大纲_公式渲染版.pdf",
                    "无人机通信劣化-论文级报告.docx",
                    "无人机通信劣化-论文级报告.pdf",
                    "无人机通信劣化_极简报告.pptx",
                ),
                ignore_globs=("~$*",),
            ),
            CopyTask(
                label="无人机研究_专题论文与参考资料",
                src=r"D:\无人机通信系统抗干扰性能的测试评估技术研究",
                dest_rel=r"参考文献\论文文献\无人机通信测试评估技术研究_专题论文与参考资料",
                task_type="参考",
                summary="该节点汇总无人机测试评估技术研究目录中的专题论文、标准、压缩包与改进方向资料，是论文无人机线新增的专题参考资料库。",
                writing_value="可直接用于补充综述、路径损耗建模、干扰机理、测试标准和改进方向论证。",
                entries=("论文", "1220论文", "1222论文", "改进"),
            ),
            CopyTask(
                label="无人机研究_测试标定与实验建设资料",
                src=r"D:\无人机通信系统抗干扰性能的测试评估技术研究",
                dest_rel=r"参考文献\研究方案与标定\无人机通信测试评估技术研究_测试标定与实验建设",
                task_type="参考",
                summary="该节点保留无人机测试评估技术研究目录中的干扰源标定和实验室建设资料，可作为论文无人机线的测试依据与实验条件说明来源。",
                writing_value="适合用于说明参数标定依据、实验环境构建和测试评估技术路线。",
                entries=("干扰源数据标定", "实验室建设"),
                ignore_globs=("~$*",),
            ),
        ],
    ),
    ProjectSpec(
        name="正向设计规范",
        description="放置正向设计规范项目，重点保留研究方案、标准规范、模板案例与总结报告。",
        writing_summary=(
            "这一条线主要承担规范化与工程化表达。研究方案目录提供总体设计方案、标准、对照文件和研究报告，模板参考目录提供可借鉴的 EMC 正向设计案例，"
            "总结报告则补充项目层面的论证背景；本次新增的一汽根目录方案定稿和无人机测试评估资料，又补齐了简版方案、项目总结、实验室建设和测试标定素材。"
            "后续无论写规范、研究方案还是论文背景，都可以直接从这里提取结构化语言。"
        ),
        cross_summary="它是卫星相关三条线的规范约束与写作支点，为 `专利自有`、`专利一汽`、`论文卫星` 提供统一的标准和方案语言。",
        tasks=[
            CopyTask(
                label="车载低轨卫星通信系统EMC正向设计研究方案",
                src=r"D:\一汽项目\车载低轨卫星通信系统EMC性能正向设计技术研究方案",
                dest_rel=r"成果本身\规范方案\车载低轨卫星通信系统EMC性能正向设计技术研究方案",
                task_type="成果",
                summary="该目录包含正向设计规范项目的研究方案、标准、对照文件与专题论文，是本条线的核心成果。",
                writing_value="可直接用于规范、方案、项目报告与论文背景撰写。",
            ),
            CopyTask(
                label="EMC正向设计写作模板参考",
                src=r"D:\一汽项目\EMC正向设计写作模板参考",
                dest_rel=r"参考文献\模板案例\EMC正向设计写作模板参考",
                task_type="参考",
                summary="该目录整理了多篇 EMC 正向设计参考案例，是正向设计规范线的重要模板资源。",
                writing_value="适合借鉴结构、术语、表述方式与章节安排。",
            ),
            CopyTask(
                label="项目总结报告",
                src=r"D:\一汽项目\技术开发项目总结报告-电动智能网联汽车电磁兼容仿真验证技术研究.docx",
                dest_rel=r"参考文献\技术报告\技术开发项目总结报告-电动智能网联汽车电磁兼容仿真验证技术研究.docx",
                task_type="参考",
                summary="该报告从项目层面总结电动智能网联汽车电磁兼容仿真验证技术研究，是正向设计规范线的背景与论证材料。",
                writing_value="可为项目意义、工程背景和综合结论部分提供成熟表述。",
            ),
            CopyTask(
                label="一汽项目_根目录方案简版与定稿",
                src=r"D:\一汽项目",
                dest_rel=r"成果本身\规范方案\一汽项目_根目录方案简版与定稿",
                task_type="成果",
                summary="该节点汇总一汽项目根目录中的方案定稿、一页纸与两页纸版本，是正向设计规范线补齐不同交付粒度的重要成果位。",
                writing_value="适合在规范、方案、汇报摘要与对外沟通版本之间快速切换内容粒度。",
                entries=(
                    "车载低轨卫星通信系统EMC性能正向设计技术研究方案.docx",
                    "车载低轨卫星通信系统EMC性能正向设计技术研究方案（两页纸简化版）.docx",
                    "车载低轨卫星通信系统EMC性能正向设计研究方案（一页纸简化版）.docx",
                    "（两页纸简化版）车载低轨卫星通信系统EMC性能正向设计技术研究方案.docx",
                    "（一页纸简化版）车载低轨卫星通信系统EMC性能正向设计研究方案.docx",
                    "（一页纸简化版）车载低轨卫星通信系统EMC性能正向设计研究方案 - 副本.docx",
                    "最终稿_格式对齐_车载低轨卫星通信系统EMC性能正向设计技术研究方案.docx",
                ),
            ),
            CopyTask(
                label="一汽项目_项目总结与学位论文材料",
                src=r"D:\一汽项目",
                dest_rel=r"参考文献\技术报告\项目总结与学位论文材料",
                task_type="参考",
                summary="该节点收纳一汽项目中的项目总结报告与学位论文终稿，可作为正向设计规范线的背景论证和长文献参考材料。",
                writing_value="适合提炼项目意义、研究现状和较完整的论证表达。",
                entries=("00 技术开发项目总结报告.docx", "技术开发项目总结报告-电动智能网联汽车电磁兼容仿真验证技术研究.docx", "16111047-宋亚丽-博士学位论文终稿.doc"),
            ),
            CopyTask(
                label="一汽项目_标准与对标参考合集",
                src=r"D:\一汽项目",
                dest_rel=r"参考文献\标准与对标\一汽项目参考文献合集",
                task_type="参考",
                summary="该节点汇总一汽项目根目录中的标准、车载卫星通信资料和参考文献目录，是正向设计规范线补充标准与对标依据的集合位。",
                writing_value="可用于补充标准引用、术语定义、竞品参考和技术边界说明。",
                entries=("参考文献", "车载卫星通信系统"),
            ),
            CopyTask(
                label="一汽项目_试验室仿真技术简版资料",
                src=r"D:\一汽项目",
                dest_rel=r"参考文献\技术报告\一汽项目试验室仿真技术简版资料",
                task_type="参考",
                summary="该节点收纳一汽项目根目录中的试验室仿真技术一页纸、两页纸和格式整理稿，可作为规范线的补充简版说明材料。",
                writing_value="适合在规范编制、阶段汇报和方案摘要中复用已有简版技术文字。",
                entries=(
                    "试验室环境下低轨卫星通信系统EMC仿真技术.docx",
                    "试验室环境下低轨卫星通信系统EMC仿真技术（一页纸简化版）.docx",
                    "试验室环境下低轨卫星通信系统EMC仿真技术（两页纸简化版）.docx",
                    "（一页纸）试验室环境下低轨卫星通信系统EMC仿真技术.docx",
                    "（一页纸）试验室环境下低轨卫星通信系统EMC仿真技术(1).docx",
                    "（两页纸）试验室环境下低轨卫星通信系统EMC仿真技术.docx",
                    "lab_emc_sim_one_page.docx",
                    "lab_emc_sim_two_page.docx",
                    "最终稿_格式对齐_试验室环境下低轨卫星通信系统EMC仿真技术.docx",
                ),
            ),
            CopyTask(
                label="无人机研究_测试评估方法与实验室建设资料",
                src=r"D:\无人机通信系统抗干扰性能的测试评估技术研究",
                dest_rel=r"参考文献\测试评估与实验室建设\无人机通信抗干扰测试评估技术研究",
                task_type="参考",
                summary="该节点汇总无人机测试评估技术研究目录中的测试标准、干扰源标定、实验室建设和论文级方法报告，是正向设计规范线吸收无人机测试评估思路的入口。",
                writing_value="可用于借鉴测试评估逻辑、实验室建设框架和参数标定口径。",
                entries=(
                    "1222论文",
                    "干扰源数据标定",
                    "实验室建设",
                    "无人机通信劣化-论文大纲_公式渲染版.docx",
                    "无人机通信劣化-论文大纲_公式渲染版.pdf",
                    "无人机通信劣化-论文级报告.docx",
                    "无人机通信劣化-论文级报告.pdf",
                ),
                ignore_globs=("~$*",),
            ),
        ],
    ),
]


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, List[Dict]] = {"generated_at": [datetime.now().isoformat(timespec="seconds")]}

    for spec in PROJECTS:
        project_root = project_root_for(spec.name)
        build_project_structure(project_root)
        results: List[TaskResult] = []
        for task in spec.tasks:
            print(f"[COPY] {spec.name} :: {task.label}")
            results.append(copy_task(project_root, task))
        build_project_vault(project_root, spec, results)
        write_update_log(project_root, spec, results)
        manifest[spec.name] = [asdict(x) for x in results]

    build_total_vault(PROJECTS)
    write_text(TOTAL_VAULT / "90_自动索引" / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"Engineering overview created at: {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
