from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import textwrap
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None


ROOT = Path(__file__).resolve().parents[2]
VAULT = ROOT / "obsidian_vault"
RESULT_DIR = VAULT / "05_成果图谱"
INDEX_DIR = VAULT / "10_总索引"
THEME_DIR = VAULT / "15_主题图谱"
CARD_DIR = VAULT / "20_知识卡片"
CHAT_DIR = VAULT / "30_对话沉淀"
AUTO_DIR = VAULT / "90_自动索引"

TEXT_EXTS = {".csv", ".json", ".md", ".ps1", ".py", ".svg", ".txt", ".yml", ".yaml", ".ipynb"}
TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk")

UAV_ROOT = ROOT / "key reference"
UAV_DRAFT_ROOT = ROOT / "论文"
SATELLITE_PAPER_ROOT = Path(r"D:\论文卫星")
UAV_PAPER_ROOT = Path(r"D:\论文无人机")
PATENT_ROOTS = [Path(r"D:\专利一汽"), Path(r"D:\专利自有")]
SIM_ROOT = Path(r"D:\项目\Opnet仿真")
DESIGN_TEMPLATE_ROOT = Path(r"D:\一汽项目\EMC正向设计写作模板参考")
DESIGN_SCHEME_ROOT = Path(r"D:\一汽项目\车载低轨卫星通信系统EMC性能正向设计技术研究方案")

CATEGORY_ORDER = ["论文无人机", "论文卫星", "专利卫星", "专利仿真", "正向设计规范"]

CATEGORY_DESC = {
    "论文无人机": "无人机通信、轨迹优化、抗干扰和评估方法相关论文与写作资料。",
    "论文卫星": "卫星通信、电磁兼容、低轨卫星车载应用相关论文资料。",
    "专利卫星": "卫星相关专利撰写资料、交底模板、专利支撑教程和对照材料。",
    "专利仿真": "用于专利或方案支撑的 OPNET/CST/网络仿真工程、案例和结果文件。",
    "正向设计规范": "围绕 EMC 正向设计的标准、方案、模板和规范性资料。",
}


@dataclass
class KnowledgeRecord:
    source_path: str
    source_name: str
    extension: str
    size_bytes: int
    modified: str
    category: str
    role: str
    source_root: str
    note_name: str
    note_path: str


def reset_vault() -> None:
    VAULT.mkdir(parents=True, exist_ok=True)
    for name in [
        "00_Home.md",
        "00_知识库总览.md",
        "05_成果图谱",
        "10_Folder_Maps",
        "20_File_Cards",
        "30_Conversation_Notes",
        "90_Auto",
        "10_总索引",
        "15_主题图谱",
        "20_知识卡片",
        "30_对话沉淀",
        "90_自动索引",
    ]:
        target = VAULT / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()
    for path in [RESULT_DIR, INDEX_DIR, THEME_DIR, CARD_DIR, CHAT_DIR, AUTO_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def path_startswith(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def existing_roots() -> List[Tuple[str, Path]]:
    roots = [
        ("无人机参考文献", UAV_ROOT),
        ("无人机论文草稿", UAV_DRAFT_ROOT),
        ("无人机下载文献", UAV_PAPER_ROOT),
        ("卫星下载文献", SATELLITE_PAPER_ROOT),
        ("卫星专利资料", Path(r"D:\专利一汽")),
        ("自有专利资料", Path(r"D:\专利自有")),
        ("仿真工程资料", SIM_ROOT),
        ("正向设计模板参考", DESIGN_TEMPLATE_ROOT),
        ("正向设计研究方案", DESIGN_SCHEME_ROOT),
    ]
    return [(label, root) for label, root in roots if root.exists()]


def iter_source_files() -> Iterable[Tuple[str, Path]]:
    seen: set[str] = set()
    for label, root in existing_roots():
        if root.is_file():
            paths = [root]
        else:
            paths = sorted(p for p in root.rglob("*") if p.is_file())
        for path in paths:
            if path.name.startswith("~$"):
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            yield label, path


def fmt_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def sanitize_name(text: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", text.strip())
    value = re.sub(r"\s+", "_", value)
    return value[:80] or "未命名"


def role_for_uav(path: Path) -> str:
    s = str(path)
    if "无人机通信组网总体背景" in s:
        return "总体背景"
    if "相关类似研究" in s:
        return "相关研究"
    if "仿真底座证据链" in s:
        return "仿真证据链"
    if "写作模版" in s:
        return "写作参考"
    if "通信评估公式方法_底层变量的影响思路" in s:
        return "通信评估方法"
    if "高度、路损、干扰耦合考虑" in s:
        return "链路耦合分析"
    if path.suffix.lower() == ".docx":
        return "论文草稿"
    return "无人机参考资料"


def role_for_patent(path: Path) -> str:
    name = path.name
    if "模板" in name:
        return "专利交底模板"
    if "培训教程" in name or "CST" in name.upper():
        return "专利支撑仿真教程"
    if "对照" in name:
        return "专利标准对照"
    return "卫星专利资料"


def role_for_sim(path: Path) -> str:
    s = str(path)
    ext = path.suffix.lower()
    if "移动无线网络仿真" in s:
        base = "无线网络建模案例"
    elif "移动无线仿真" in s:
        base = "移动无线仿真案例"
    else:
        base = "仿真练习资料"
    if ext in {".prj", ".seq", ".nd.m", ".pa.m", ".pb.m", ".pr.m", ".cml", ".trj"}:
        return f"{base}_工程文件"
    if ext in {".gif", ".png", ".jpg", ".jpeg", ".ov", ".ot", ".ef", ".desinfo"}:
        return f"{base}_结果产物"
    if ext == ".zip":
        return f"{base}_压缩包"
    return base


def role_for_design(path: Path) -> str:
    s = str(path)
    name = path.name
    if "\\标准\\" in s or "技术条件" in name or "要求和测量方法" in name or "规范" in name:
        return "标准规范"
    if "总体设计方案" in name or "研究方案" in name or "初版" in name or "最终稿" in name:
        return "研究方案"
    if "总EMC相关" in name:
        return "EMC汇总资料"
    return "模板参考案例"


def classify_file(source_root: str, path: Path) -> Optional[Tuple[str, str]]:
    if path_startswith(path, UAV_ROOT) or path_startswith(path, UAV_DRAFT_ROOT) or path_startswith(path, UAV_PAPER_ROOT):
        return "论文无人机", role_for_uav(path)

    if any(path_startswith(path, root) for root in PATENT_ROOTS):
        return "专利卫星", role_for_patent(path)

    if path_startswith(path, SIM_ROOT):
        return "专利仿真", role_for_sim(path)

    if path_startswith(path, DESIGN_SCHEME_ROOT) and "\\论文\\" in str(path):
        return "论文卫星", "卫星参考论文"

    if path_startswith(path, SATELLITE_PAPER_ROOT):
        return "论文卫星", "卫星下载文献"

    if path_startswith(path, DESIGN_TEMPLATE_ROOT):
        if path.suffix.lower() == ".pdf" and any(keyword in path.name for keyword in ["卫星", "空间站"]):
            return "论文卫星", "卫星参考论文"
        return "正向设计规范", "模板参考案例"

    if path_startswith(path, DESIGN_SCHEME_ROOT):
        if "专利与标准对照" in path.name:
            return "专利卫星", "专利标准对照"
        return "正向设计规范", role_for_design(path)

    if source_root == "卫星下载文献":
        return "论文卫星", "卫星下载文献"
    return None


def read_text(path: Path, max_chars: int = 2000) -> str:
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)[:max_chars]
        except Exception:
            continue
    return ""


def extract_docx_preview(path: Path, max_paragraphs: int = 16) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            data = zf.read("word/document.xml")
    except Exception:
        return ""
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return ""
    texts = []
    for node in root.iter():
        if node.tag.endswith("}t") and node.text:
            value = node.text.strip()
            if value:
                texts.append(value)
    return "\n".join(texts[:max_paragraphs])


def extract_pdf_preview(path: Path, max_chars: int = 1200) -> str:
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


def extract_zip_preview(path: Path, max_items: int = 20) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            return "\n".join(zf.namelist()[:max_items])
    except Exception as exc:
        return f"压缩包预览失败: {exc}"


def preview_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        return read_text(path)
    if ext == ".docx":
        return extract_docx_preview(path)
    if ext == ".pdf":
        return extract_pdf_preview(path)
    if ext == ".zip":
        return extract_zip_preview(path)
    return ""


def collect_records() -> List[KnowledgeRecord]:
    role_counter: Dict[Tuple[str, str], int] = defaultdict(int)
    records: List[KnowledgeRecord] = []
    for source_root, path in iter_source_files():
        classified = classify_file(source_root, path)
        if classified is None:
            continue
        category, role = classified
        role_counter[(category, role)] += 1
        idx = role_counter[(category, role)]
        note_name = f"{category}_{role}_{idx:03d}"
        note_path = CARD_DIR / category / role / f"{sanitize_name(note_name)}.md"
        records.append(
            KnowledgeRecord(
                source_path=str(path),
                source_name=path.name,
                extension=path.suffix.lower(),
                size_bytes=path.stat().st_size,
                modified=fmt_mtime(path),
                category=category,
                role=role,
                source_root=source_root,
                note_name=note_name,
                note_path=str(note_path),
            )
        )
    records.sort(key=lambda r: (CATEGORY_ORDER.index(r.category), r.role, r.note_name))
    return records


def rel_link(from_path: Path, to_path: Path) -> str:
    return os.path.relpath(to_path, from_path.parent).replace("\\", "/")


def category_note_path(category: str) -> Path:
    return INDEX_DIR / f"{sanitize_name(category)}.md"


def role_note_path(category: str, role: str) -> Path:
    return THEME_DIR / category / f"{sanitize_name(role)}.md"


def build_home(records: Sequence[KnowledgeRecord]) -> str:
    counts = defaultdict(int)
    for record in records:
        counts[record.category] += 1
    lines = [
        "# 知识库总览",
        "",
        "本知识库只改变 Obsidian 中的中文逻辑展示，不改动任何原始文件或原始目录名。",
        "",
        f"- 生成时间：`{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- 知识卡片总数：`{len(records)}`",
        "",
        "## 五大知识项",
        "",
    ]
    current = VAULT / "00_知识库总览.md"
    lines.append(f"- [成果知识图谱]({rel_link(current, RESULT_DIR / '00_成果总览.md')})：按你的实际成果组织 Python 无人机模型、MATLAB 卫星模型、专利仿真支撑材料。")
    for category in CATEGORY_ORDER:
        target = category_note_path(category)
        lines.append(f"- [{category}]({rel_link(current, target)})：{CATEGORY_DESC[category]}，当前 ` {counts.get(category, 0)} ` 项")
    lines.extend(
        [
            "",
            "## 使用方式",
            "",
            "- 刷新整个知识库：`python tools\\obsidian\\build_obsidian_vault.py`",
            "- 写入一次对话沉淀：`python tools\\obsidian\\capture_knowledge_note.py --title \"中文标题\" --stdin`",
            "- 直接在 Codex 里要求“写入知识库”“更新某类图谱”“重建 Obsidian 索引”，我会继续维护这个 vault。",
        ]
    )
    return "\n".join(lines)


def build_category_note(category: str, records: Sequence[KnowledgeRecord]) -> str:
    role_groups: Dict[str, List[KnowledgeRecord]] = defaultdict(list)
    for record in records:
        if record.category == category:
            role_groups[record.role].append(record)

    current = category_note_path(category)
    lines = [
        f"# {category}",
        "",
        CATEGORY_DESC[category],
        "",
    ]
    if not role_groups:
        lines.extend(
            [
                "当前没有检测到可纳入该分类的实际文件。",
                "",
                "可能原因：原始目录为空、只有快捷方式、或资料尚未放入已监控目录。",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(["## 主题分组", ""])
    for role in sorted(role_groups.keys()):
        target = role_note_path(category, role)
        lines.append(f"- [{role}]({rel_link(current, target)})：`{len(role_groups[role])}` 项")
    lines.append("")
    lines.extend(["## 来源目录", ""])
    roots = sorted({r.source_root for r in records if r.category == category})
    for root in roots:
        lines.append(f"- `{root}`")
    lines.append("")
    return "\n".join(lines)


def build_role_note(category: str, role: str, records: Sequence[KnowledgeRecord]) -> str:
    current = role_note_path(category, role)
    lines = [
        f"# {role}",
        "",
        f"- 所属大项：[{category}]({rel_link(current, category_note_path(category))})",
        f"- 卡片数量：`{len(records)}`",
        "",
        "## 知识卡片",
        "",
    ]
    for record in records:
        target = Path(record.note_path)
        lines.append(f"- [{record.note_name}]({rel_link(current, target)})")
    lines.append("")
    return "\n".join(lines)


def build_card(record: KnowledgeRecord) -> str:
    current = Path(record.note_path)
    path = Path(record.source_path)
    preview = preview_text(path)
    lines = [
        "---",
        f'title: "{record.note_name}"',
        f'category: "{record.category}"',
        f'role: "{record.role}"',
        f'source_path: "{record.source_path}"',
        f'source_root: "{record.source_root}"',
        f'modified: "{record.modified}"',
        f"size_bytes: {record.size_bytes}",
        "---",
        f"# {record.note_name}",
        "",
        f"- 所属大项：[{record.category}]({rel_link(current, category_note_path(record.category))})",
        f"- 主题节点：[{record.role}]({rel_link(current, role_note_path(record.category, record.role))})",
        f"- 原始文件名：`{record.source_name}`",
        f"- 原始路径：`{record.source_path}`",
        f"- 来源目录标签：`{record.source_root}`",
        f"- 修改时间：`{record.modified}`",
        f"- 文件大小：`{record.size_bytes}` bytes",
        "",
        "## 原始标题",
        "",
        f"`{path.stem}`",
        "",
    ]
    if preview:
        lines.extend(["## 内容预览", "", "```text", preview[:2000], "```", ""])
    else:
        lines.extend(["## 内容预览", "", "当前文件类型未提取文本预览，但已保留原始路径与逻辑分类。", ""])
    return "\n".join(lines)


def build_chat_readme() -> str:
    return textwrap.dedent(
        """
        # 对话沉淀

        这里用于保存 Codex 对话结论、实验摘要、文献摘记和阶段性判断。

        示例：

        ```powershell
        @'
        这里写入本次对话的中文总结。
        '@ | python tools/obsidian/capture_knowledge_note.py --title "阶段总结" --tags 对话沉淀,知识更新 --stdin
        ```
        """
    ).strip()


def link_or_path(record_by_source: Dict[str, KnowledgeRecord], source_path: str) -> str:
    record = record_by_source.get(source_path)
    if record:
        return rel_link(RESULT_DIR / "00_成果总览.md", Path(record.note_path))
    return source_path


def source_key(source_path: str) -> str:
    return Path(source_path).as_posix()


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="gb18030")
        except Exception:
            return ""


def parse_corner_case_summary() -> Dict[str, str]:
    summary_path = ROOT / "12" / "output" / "measured_corner_compare" / "corner_case_summary.json"
    if not summary_path.exists():
        return {}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, str] = {}
    for key in ["aerpaw", "dryad"]:
        try:
            grp = data["datasets"][key]["groups"]
            out[f"{key}_measured_mean_sinr"] = f"{grp['measured_trajectory']['sinr_db']['mean']:.2f} dB"
            out[f"{key}_worst_mean_sinr"] = f"{grp['model_worst_region']['sinr_db']['mean']:.2f} dB"
            out[f"{key}_worst_outage_10"] = f"{100*grp['model_worst_region']['outage']['10.0']:.0f}%"
        except Exception:
            continue
    return out


def parse_leo_summary() -> Dict[str, str]:
    summary_path = Path(r"D:\一汽项目\LEO_Sim\LEO_Sim_V7_modified\outputs_v7\summary_v7.txt")
    if not summary_path.exists():
        return {}
    text = safe_read_text(summary_path)
    out: Dict[str, str] = {}
    patterns = {
        "dl_base": r"DL Base\s*: meanThr = ([\d\.]+ Mbps), outage = ([\d\.]+ %)",
        "dl_worst": r"DL Worst\s*: meanThr = ([\d\.]+ Mbps), outage = ([\d\.]+ %)",
        "ul_base": r"UL Base\s*: meanThr = ([\d\.]+ Mbps), outage = ([\d\.]+ %)",
        "ul_worst": r"UL Worst\s*: meanThr = ([\d\.]+ Mbps), outage = ([\d\.]+ %)",
        "e2e_base": r"E2E Base\s*: meanThr = ([\d\.]+ Mbps), outage = ([\d\.]+ %)",
        "e2e_worst": r"E2E Worst\s*: meanThr = ([\d\.]+ Mbps), outage = ([\d\.]+ %)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            out[key] = f"吞吐 {match.group(1)}，中断率 {match.group(2)}"
    return out


def outcome_note(path: Path, title: str, body: str) -> None:
    write_text(
        path,
        "\n".join(
            [
                "---",
                f'title: "{title}"',
                "---",
                f"# {title}",
                "",
                body.strip(),
            ]
        ),
    )


def build_outcome_graph(records: Sequence[KnowledgeRecord]) -> None:
    records_by_source = {record.source_path: record for record in records}
    records_by_norm = {source_key(record.source_path): record for record in records}
    corner = parse_corner_case_summary()
    leo = parse_leo_summary()

    def note_link(current: Path, source_path: str) -> str:
        record = records_by_norm.get(source_key(source_path))
        if record is None:
            return Path(source_path).as_posix()
        return rel_link(current, Path(record.note_path))

    result_root = RESULT_DIR / "节点"
    achievement_dir = result_root / "成果节点"
    source_dir = result_root / "源文件节点"
    evidence_dir = result_root / "证据节点"
    for path in [achievement_dir, source_dir, evidence_dir]:
        path.mkdir(parents=True, exist_ok=True)

    overview = RESULT_DIR / "00_成果总览.md"
    achievement_uav = achievement_dir / "论文无人机_成果节点.md"
    achievement_sat = achievement_dir / "论文卫星_成果节点.md"
    achievement_patent = achievement_dir / "专利仿真_成果节点.md"

    src_uav_core = source_dir / "论文无人机_核心场景与链路预算.md"
    src_uav_pipeline = source_dir / "论文无人机_GA_GAN与评估流程.md"
    src_uav_output = source_dir / "论文无人机_实测对照与输出结果.md"
    src_sat_core = source_dir / "论文卫星_MATLAB工程主程序.md"
    src_sat_cfg = source_dir / "论文卫星_参数与链路配置.md"
    src_sat_out = source_dir / "论文卫星_交付结果与说明报告.md"
    src_patent_docs = source_dir / "专利仿真_交底书与附图清单.md"
    src_patent_drawings = source_dir / "专利仿真_附图与压缩包.md"
    src_patent_tpl = source_dir / "专利仿真_交底模板材料.md"
    src_patent_cst = source_dir / "专利仿真_CST仿真教程.md"

    ev_uav_basis = evidence_dir / "证据_无人机综述与论文草稿.md"
    ev_uav_method = evidence_dir / "证据_无人机优化与评估依据.md"
    ev_uav_measure = evidence_dir / "证据_无人机实测与仿真底座.md"
    ev_sat_papers = evidence_dir / "证据_卫星EMC参考论文.md"
    ev_sat_plan = evidence_dir / "证据_车载低轨卫星研究方案.md"
    ev_sat_std = evidence_dir / "证据_卫星EMC规范与对照.md"

    outcome_note(
        overview,
        "成果总览",
        f"""
本页按你的实际成果重新组织知识图谱，不再把“论文无人机/论文卫星/专利仿真”仅仅当成资料分类，而是当成成果节点。

## 节点总结

- `论文无人机`：对应 [Python 无人机模型]({rel_link(overview, achievement_uav)})，核心是 `12/` 下的 GA+GAN 场景生成、链路预算、KPI/BLER 评估与测量对照。
- `论文卫星`：对应 [MATLAB 卫星模型]({rel_link(overview, achievement_sat)})，桌面 `D:\\论文卫星` 当前没有实际工程文件，真实成果位于 `D:\\一汽项目\\LEO_Sim` 与 `D:\\一汽项目\\satellite.m`。
- `专利仿真`：对应 [专利一汽支撑材料]({rel_link(overview, achievement_patent)})，当前位于 `D:\\专利一汽`，已经形成交底书、附图清单、10 张专利附图、两个压缩包和 CST 教程，适合作为专利撰写与仿真支撑包。

## 图谱入口

- [论文无人机 成果节点]({rel_link(overview, achievement_uav)})
- [论文卫星 成果节点]({rel_link(overview, achievement_sat)})
- [专利仿真 成果节点]({rel_link(overview, achievement_patent)})
""",
    )

    outcome_note(
        achievement_uav,
        "论文无人机 成果节点",
        f"""
## 节点总结

这是一个基于 Python 的无人机通信性能评估模型，围绕“最差通信场景”展开，核心机制是 `GA 采样 + GAN 生成 + KPI/BLER 评估 + 实测 corner-case 对照`。论文草稿已经明确写出题目为“基于遗传算法与生成对抗网络的无人机通信性能评估模型”。桌面 `D:\\论文无人机` 当前没有实际工程文件，真实模型位于当前仓库 `D:\\UAV_Communication_GA\\12`。

## 成果结构

- [核心场景与链路预算]({rel_link(achievement_uav, src_uav_core)})
- [GA/GAN 与评估流程]({rel_link(achievement_uav, src_uav_pipeline)})
- [实测对照与输出结果]({rel_link(achievement_uav, src_uav_output)})

## 证据链

- [证据：无人机综述与论文草稿]({rel_link(achievement_uav, ev_uav_basis)})
- [证据：无人机优化与评估依据]({rel_link(achievement_uav, ev_uav_method)})
- [证据：无人机实测与仿真底座]({rel_link(achievement_uav, ev_uav_measure)})

## 当前可直接定位的原始成果位置

- `D:\\UAV_Communication_GA\\12\\UAV_GA.py`
- `D:\\UAV_Communication_GA\\12\\gan_uav_pipeline.py`
- `D:\\UAV_Communication_GA\\12\\compare_random_ga_gan.py`
- `D:\\UAV_Communication_GA\\12\\evaluate.py`
- `D:\\UAV_Communication_GA\\12\\PROJECT_GUIDE.md`
- `D:\\UAV_Communication_GA\\12\\COMM_DEGRADATION_REPORT.md`
- `D:\\UAV_Communication_GA\\12\\output\\measured_corner_compare\\corner_case_summary.json`
""",
    )

    outcome_note(
        achievement_sat,
        "论文卫星 成果节点",
        f"""
## 节点总结

这是一个基于 MATLAB 的车载低轨卫星 EMC 工程模型，主线成果位于 `D:\\一汽项目\\LEO_Sim`。桌面 `D:\\论文卫星` 当前没有实际 MATLAB 工程内容，真实模型以 `LEO_StarNet_EMC_V7_0_Engineering.m`、`emcDefaultConfig.m`、`summary_v7.txt` 等文件为主。模型包含星座构建、上下行链路、干扰机建模、最坏工况搜索、合规性检查和结果摘要输出，已经形成 V7 工程交付版本。

## 成果结构

- [MATLAB 工程主程序]({rel_link(achievement_sat, src_sat_core)})
- [参数与链路配置]({rel_link(achievement_sat, src_sat_cfg)})
- [交付结果与说明报告]({rel_link(achievement_sat, src_sat_out)})

## 证据链

- [证据：卫星 EMC 参考论文]({rel_link(achievement_sat, ev_sat_papers)})
- [证据：车载低轨卫星研究方案]({rel_link(achievement_sat, ev_sat_plan)})
- [证据：卫星 EMC 规范与对照]({rel_link(achievement_sat, ev_sat_std)})

## 当前可直接定位的原始成果位置

- `D:\\一汽项目\\LEO_Sim\\run_LEO_EMC_Sim.m`
- `D:\\一汽项目\\LEO_Sim\\LEO_Sim_V7_modified\\v7proj\\LEO_StarNet_EMC_V7_0_Engineering.m`
- `D:\\一汽项目\\LEO_Sim\\LEO_Sim_V7_modified\\v7proj\\emcDefaultConfig.m`
- `D:\\一汽项目\\LEO_Sim\\LEO_Sim_V7_modified\\v7proj\\emcBuildLinkModel.m`
- `D:\\一汽项目\\LEO_Sim\\LEO_Sim_V7_modified\\outputs_v7\\summary_v7.txt`
- `D:\\一汽项目\\LEO_Sim\\LEO_EMC_Sim_说明报告.docx`
- `D:\\一汽项目\\LEO_Sim\\LEO_EMC_仿真技术_主流方法与文献支撑_说明报告.docx`
""",
    )

    outcome_note(
        achievement_patent,
        "专利仿真 成果节点",
        f"""
## 节点总结

你当前称为“专利仿真”的成果，实际存放在 `D:\\专利一汽`。我检查到里面已经形成四层内容：交底模板、第三版交底书与附图清单、10 张专利附图及其压缩包、CST MWS 教程。也就是说，这个节点当前更接近“专利撰写与仿真支撑包”，它依托于卫星 MATLAB 模型与 EMC 研究方案，但本身不是一个独立的求解工程目录。

## 成果结构

- [交底书与附图清单]({rel_link(achievement_patent, src_patent_docs)})
- [附图与压缩包]({rel_link(achievement_patent, src_patent_drawings)})
- [交底模板材料]({rel_link(achievement_patent, src_patent_tpl)})
- [CST 仿真教程]({rel_link(achievement_patent, src_patent_cst)})

## 相关证据与依托

- [证据：车载低轨卫星研究方案]({rel_link(achievement_patent, ev_sat_plan)})
- [证据：卫星 EMC 规范与对照]({rel_link(achievement_patent, ev_sat_std)})
- [证据：卫星 EMC 参考论文]({rel_link(achievement_patent, ev_sat_papers)})
- [论文卫星 成果节点]({rel_link(achievement_patent, achievement_sat)})

## 当前可直接定位的原始成果位置

- `D:\\专利一汽\\20260318_交底书_第三版_一种卫星通信天线的实验室环境模拟测试系统及方法.docx`
- `D:\\专利一汽\\20260318_附图清单_第三版_卫星通信天线的实验室环境模拟测试系统_重新生成可下载版.docx`
- `D:\\专利一汽\\20260318_附图清单_优化重绘版_可下载.docx`
- `D:\\专利一汽\\20260318_Visio附图_卫星通信天线实验室环境模拟测试系统_10图_SVG_PNG.zip`
- `D:\\专利一汽\\20260318_专利附图_优化重绘版_SVG_PNG.zip`
- `D:\\专利一汽\\20250912_交底书模板_发明及实用新型.docx`
- `D:\\专利一汽\\CST_MWS_培训教程－初级cn.pdf`
- `D:\\专利一汽\\CST MWS 培训教程－中级.pdf`
""",
    )

    outcome_note(
        src_uav_core,
        "论文无人机 核心场景与链路预算",
        f"""
## 节点总结

该源文件节点对应 Python 无人机模型的物理与优化底座。`UAV_GA.py` 中定义了 `DroneCommProblem`、场景生成、干扰源配置、接收功率/干扰噪声/功率裕量计算，以及综合劣化指标；`COMM_DEGRADATION_REPORT.md` 对公式作了通俗解释。

## 原始文件位置

- `D:\\UAV_Communication_GA\\12\\UAV_GA.py`
- `D:\\UAV_Communication_GA\\12\\COMM_DEGRADATION_REPORT.md`
- `D:\\UAV_Communication_GA\\12\\INTERFERENCE_SOURCE_TABLE.md`

## 关键内容

- `DroneCommProblem` 是核心优化问题定义，负责城市、无人机、干扰源和链路约束建模。
- 通信劣化采用“功率裕量 -> S 型映射”的思路，速度/能效劣化单独建模，再组合成总劣化。
- 干扰源表已经细化到 WiFi/4G/5G/GNSS 干扰机/工业设备/卫星地面源/蜂窝 UE 上行等类型。
""",
    )

    outcome_note(
        src_uav_pipeline,
        "论文无人机 GA GAN 与评估流程",
        f"""
## 节点总结

该节点对应无人机模型的实验流程层。`gan_uav_pipeline.py` 用 GA 样本训练 GAN 并生成场景，`compare_random_ga_gan.py` 对比 GA/GAN/Random，`evaluate.py` 负责 KPI、BLER 与吞吐评估。

## 原始文件位置

- `D:\\UAV_Communication_GA\\12\\gan_uav_pipeline.py`
- `D:\\UAV_Communication_GA\\12\\compare_random_ga_gan.py`
- `D:\\UAV_Communication_GA\\12\\evaluate.py`
- `D:\\UAV_Communication_GA\\12\\PROJECT_GUIDE.md`

## 关键内容

- `gan_uav_pipeline.py` 先用 `geatpy` 采样最优或最差场景，再把样本缩放后交给 PyTorch MLP GAN。
- `compare_random_ga_gan.py` 把 `GA / GAN / Random` 放到同一评估框架里做统计与可视化对比。
- `evaluate.py` 进一步把链路记录映射成 outage、throughput、EE、BLER_A、BLER_B 指标。
""",
    )

    outcome_note(
        src_uav_output,
        "论文无人机 实测对照与输出结果",
        f"""
## 节点总结

该节点对应无人机模型已经形成的实验输出。现有结果重点体现在 `measured_corner_compare`，将模型最劣区域与 AERPAW、Dryad 的 5G 实测轨迹进行对照。

## 原始文件位置

- `D:\\UAV_Communication_GA\\12\\output\\measured_corner_compare\\corner_case_summary.json`
- `D:\\UAV_Communication_GA\\12\\output\\measured_corner_compare\\aerpaw\\corner_case_report.json`
- `D:\\UAV_Communication_GA\\12\\output\\measured_corner_compare\\dryad\\corner_case_report.json`

## 关键结果摘要

- AERPAW：实测轨迹平均 SINR 约 `{corner.get('aerpaw_measured_mean_sinr', '未解析')}`，模型最差区域平均 SINR 约 `{corner.get('aerpaw_worst_mean_sinr', '未解析')}`，10 dB 阈值 outage `{corner.get('aerpaw_worst_outage_10', '未解析')}`。
- Dryad：实测轨迹平均 SINR 约 `{corner.get('dryad_measured_mean_sinr', '未解析')}`，模型最差区域平均 SINR 约 `{corner.get('dryad_worst_mean_sinr', '未解析')}`，10 dB 阈值 outage `{corner.get('dryad_worst_outage_10', '未解析')}`。
- 这说明模型已经不只是理论推演，而是具备和公开实测数据对齐的输出层。
""",
    )

    outcome_note(
        src_sat_core,
        "论文卫星 MATLAB 工程主程序",
        f"""
## 节点总结

该节点对应 MATLAB 卫星模型的主执行层。`LEO_StarNet_EMC_V7_0_Engineering.m` 是 V7 工程版主程序，负责星座场景、上下行、干扰机、最坏工况搜索、图形与输出；`run_LEO_EMC_Sim.m` 是 Simulink 侧运行入口；`satellite.m` 则是更基础的通信链路仿真脚本。

## 原始文件位置

- `D:\\一汽项目\\LEO_Sim\\LEO_Sim_V7_modified\\v7proj\\LEO_StarNet_EMC_V7_0_Engineering.m`
- `D:\\一汽项目\\LEO_Sim\\run_LEO_EMC_Sim.m`
- `D:\\一汽项目\\satellite.m`

## 关键内容

- 主程序中明确写到是 “Engineering Delivery Version”，并包含二维星座视图、Sky View、频率复用、ISL 图、上下行 EMC 分析。
- `run_LEO_EMC_Sim.m` 用于运行 Simulink 模型并输出 `errRate=[BER,numErr,numBits]`。
- `satellite.m` 体现了更基础的通信仿真底座，包括 QPSK、AWGN、Turbo 编码和 BER 曲线。
""",
    )

    outcome_note(
        src_sat_cfg,
        "论文卫星 参数与链路配置",
        f"""
## 节点总结

该节点对应卫星 MATLAB 模型的参数层和链路层。`emcDefaultConfig.m` 给出工程默认参数，`emcBuildLinkModel.m` 将几何、功率、干扰、频点和合规阈值组装成链路模型。

## 原始文件位置

- `D:\\一汽项目\\LEO_Sim\\LEO_Sim_V7_modified\\v7proj\\emcDefaultConfig.m`
- `D:\\一汽项目\\LEO_Sim\\LEO_Sim_V7_modified\\v7proj\\emcBuildLinkModel.m`

## 关键内容

- 默认工程名就是“车载低轨卫星通信系统EMC性能正向设计技术研究 | V7.0”。
- 星座默认参数为 `12` 个轨道面、每面 `8` 星、轨道高度 `1200 km`、倾角 `53°`、频率复用因子 `4`。
- 下行中心频率 `1.5 GHz`，上行中心频率 `1.6 GHz`，并包含最坏工况搜索、分类器、合规性和 JA3700 等要求。
""",
    )

    outcome_note(
        src_sat_out,
        "论文卫星 交付结果与说明报告",
        f"""
## 节点总结

该节点对应卫星 MATLAB 模型已经落地的交付结果。`summary_v7.txt` 把基线与最坏工况的上下行和端到端吞吐/中断率写成了交付摘要，说明报告文档则承担工程说明和文献支撑说明。

## 原始文件位置

- `D:\\一汽项目\\LEO_Sim\\LEO_Sim_V7_modified\\outputs_v7\\summary_v7.txt`
- `D:\\一汽项目\\LEO_Sim\\LEO_Sim_V7\\outputs_v7\\summary_v7.txt`
- `D:\\一汽项目\\LEO_Sim\\LEO_EMC_Sim_说明报告.docx`
- `D:\\一汽项目\\LEO_Sim\\LEO_EMC_仿真技术_主流方法与文献支撑_说明报告.docx`

## 关键结果摘要

- 下行基线：{leo.get('dl_base', '未解析')}
- 下行最坏：{leo.get('dl_worst', '未解析')}
- 上行基线：{leo.get('ul_base', '未解析')}
- 上行最坏：{leo.get('ul_worst', '未解析')}
- 端到端基线：{leo.get('e2e_base', '未解析')}
- 端到端最坏：{leo.get('e2e_worst', '未解析')}
""",
    )

    outcome_note(
        src_patent_docs,
        "专利仿真 交底书与附图清单",
        f"""
## 节点总结

该节点对应 `D:\\专利一汽` 中已经成形的专利正文材料。它们把“实验室环境模拟测试系统及方法”的技术方案、指标口径、附图说明和输出页面组织成可提交的文字版成果，是专利仿真包里最接近正式交付文本的部分。

## 原始文件位置

- `D:\\专利一汽\\20260318_交底书_第三版_一种卫星通信天线的实验室环境模拟测试系统及方法.docx`
- `D:\\专利一汽\\20260318_附图清单_第三版_卫星通信天线的实验室环境模拟测试系统_重新生成可下载版.docx`
- `D:\\专利一汽\\20260318_附图清单_优化重绘版_可下载.docx`

## 关联文献 / 研究 / 报告

- [证据：车载低轨卫星研究方案]({rel_link(src_patent_docs, ev_sat_plan)})
- [证据：卫星 EMC 规范与对照]({rel_link(src_patent_docs, ev_sat_std)})
- [证据：卫星 EMC 参考论文]({rel_link(src_patent_docs, ev_sat_papers)})
- [论文卫星 成果节点]({rel_link(src_patent_docs, achievement_sat)})

## 节点判断

- 交底书第三版已经从“研究方案”进入“专利表达”阶段，重点是把仿真、换算、扫描和判定流程固定为专利术语。
- 附图清单与交底书是强耦合关系，前者负责图号、图意和页面组织，后者负责技术方案展开。
- 这一节点与研究方案、标准规范、参考论文的关系最强，因为它需要同时满足“能写成专利”“能对应标准”“有工程与文献依据”三件事。
""",
    )

    outcome_note(
        src_patent_drawings,
        "专利仿真 附图与压缩包",
        f"""
## 节点总结

该节点对应 `D:\\专利一汽` 中的图形化成果。当前已经整理出 10 张附图，并分别保存在目录、Visio 版压缩包和优化重绘版压缩包中，形成了可以直接继续修改、导出和插入专利文本的图形资产。

## 原始文件位置

- `D:\\专利一汽\\README.txt`
- `D:\\专利一汽\\图1_系统总体架构图.svg`
- `D:\\专利一汽\\图2_实验室环境模拟测试总体布局图.svg`
- `D:\\专利一汽\\图3_源与卫星通信天线空间位置关系图.svg`
- `D:\\专利一汽\\图4_空间采样区域定义图.svg`
- `D:\\专利一汽\\图5_代表场强提取示意图.svg`
- `D:\\专利一汽\\图6_场强到接收机等效输入信号换算流程图.svg`
- `D:\\专利一汽\\图7_灵敏度判定流程图.svg`
- `D:\\专利一汽\\图8_参数扫描与最不利工况输出图.svg`
- `D:\\专利一汽\\图9_替代实施方式示意图.svg`
- `D:\\专利一汽\\图10_输出报告页面示意图.svg`
- `D:\\专利一汽\\20260318_Visio附图_卫星通信天线实验室环境模拟测试系统_10图_SVG_PNG.zip`
- `D:\\专利一汽\\20260318_专利附图_优化重绘版_SVG_PNG.zip`

## 关联文献 / 研究 / 报告

- [论文卫星 参数与链路配置]({rel_link(src_patent_drawings, src_sat_cfg)})
- [论文卫星 交付结果与说明报告]({rel_link(src_patent_drawings, src_sat_out)})
- [证据：车载低轨卫星研究方案]({rel_link(src_patent_drawings, ev_sat_plan)})
- [证据：卫星 EMC 规范与对照]({rel_link(src_patent_drawings, ev_sat_std)})

## 节点判断

- 图 1 到图 10 基本覆盖了系统架构、空间关系、采样区域、场强提取、场强到接收机等效输入信号换算、灵敏度判定、最不利工况扫描与输出报告页面，已经形成完整流程链。
- `README.txt` 明确写出当前附图口径采用 `Erep`、`AF`、`Lsys`、`Kconv`、`Veq`、`Peq`、`Ssens`、`Margin`，这说明专利附图已经和参数换算链条绑定。
- 这些图与卫星 MATLAB 模型的参数节点、结果节点关系最强，因为它们本质上是在把工程分析过程重述为专利流程图和结构图。
""",
    )

    outcome_note(
        src_patent_tpl,
        "专利仿真 交底模板材料",
        f"""
## 节点总结

该节点是 `D:\\专利一汽` 中最直接的专利撰写材料。它并不产生仿真结果，而是定义专利交底书、基础信息统计表和附图原文件的提交要求，因此更适合作为专利成果包装层。

## 原始文件位置

- `D:\\专利一汽\\20250912_交底书模板_发明及实用新型.docx`

## 关键内容

- 模板明确要求同时提交《专利交底书》、基础信息统计表和可编辑附图原文件。
- 它与 [证据：卫星 EMC 规范与对照]({rel_link(src_patent_tpl, ev_sat_std)}) 关系最强，因为后者已经有“专利技术与标准条款对照文件”。
""",
    )

    outcome_note(
        src_patent_cst,
        "专利仿真 CST 仿真教程",
        f"""
## 节点总结

该节点对应 `D:\\专利一汽` 中的 CST MWS 初级/中级教程。它们是专利仿真支撑材料，作用是帮助把想法落到电磁仿真工具，而不是直接给出项目成果结论。

## 原始文件位置

- `D:\\专利一汽\\CST_MWS_培训教程－初级cn.pdf`
- `D:\\专利一汽\\CST MWS 培训教程－中级.pdf`

## 关键内容

- 这两份材料与 [论文卫星 成果节点]({rel_link(src_patent_cst, achievement_sat)}) 存在明显方法关联，因为卫星成果本身就是 EMC 工程建模。
- 它们与 [证据：卫星 EMC 参考论文]({rel_link(src_patent_cst, ev_sat_papers)}) 和 [证据：车载低轨卫星研究方案]({rel_link(src_patent_cst, ev_sat_plan)}) 共同构成“工具 + 方案 + 理论”三层支撑。
""",
    )

    outcome_note(
        ev_uav_basis,
        "证据 无人机综述与论文草稿",
        f"""
## 节点总结

这组证据解释“论文无人机成果为什么成立”。一方面有已经成文的论文初稿，另一方面有两篇综述提供研究背景、应用场景和技术趋势。

## 关联原文位置

- [论文无人机_论文草稿_002]({note_link(ev_uav_basis, "D:/UAV_Communication_GA/论文/论文初稿.docx")})
- [论文无人机_总体背景_001]({note_link(ev_uav_basis, "D:/UAV_Communication_GA/key reference/无人机通信组网总体背景/A survey on UAV-assisted wireless communications Recent advances and future trends（Computer Communication Elsevier）.pdf")})
- [论文无人机_总体背景_002]({note_link(ev_uav_basis, "D:/UAV_Communication_GA/key reference/无人机通信组网总体背景/Survey_on_UAV_Cellular_Communications_Practical_Aspects_Standardization_Advancements_Regulation_and_Security_Challenges.pdf")})

## 节点判断

- 论文草稿已经明确写出模型目标、GA+GAN 方法、SINR 与能耗双重评估底座。
- 两篇综述分别提供了 UAV-assisted wireless communications 与 UAV 蜂窝通信背景，足以作为选题和引言支撑。
""",
    )

    outcome_note(
        ev_uav_method,
        "证据 无人机优化与评估依据",
        f"""
## 节点总结

这组证据直接对应无人机 Python 模型的方法论来源：轨迹优化、最坏情况搜索和链路统计评估。

## 关联原文位置

- [论文无人机_相关研究_006]({note_link(ev_uav_method, "D:/UAV_Communication_GA/key reference/相关类似研究/Robust trajectory planning for UAV.pdf")})
- [论文无人机_通信评估方法_016]({note_link(ev_uav_method, "D:/UAV_Communication_GA/key reference/通信评估公式方法_底层变量的影响思路/On_the_Downlink_SINR_Meta_Distribution_of_UAV-Assisted_Wireless_Networks.pdf")})
- [论文无人机_无人机参考资料_005]({note_link(ev_uav_method, "D:/UAV_Communication_GA/key reference/Genetic Algorithm-Based Placement and Resource.pdf")})

## 节点判断

- `Robust trajectory planning for UAV` 对应轨迹/最坏工况求解逻辑。
- `On_the_Downlink_SINR_Meta_Distribution...` 对应链路层 SINR 与统计分布建模思路。
- `Genetic Algorithm-Based Placement and Resource` 对应遗传算法在资源/位置联合优化中的方法支撑。
""",
    )

    outcome_note(
        ev_uav_measure,
        "证据 无人机实测与仿真底座",
        f"""
## 节点总结

这组证据支撑无人机模型的“可验证性”。它既包括通信实验/PHY 抽象文献，也包括已经跑出的 AERPAW 与 Dryad 实测对照结果。

## 关联原文位置

- [论文无人机_仿真证据链_002]({note_link(ev_uav_measure, "D:/UAV_Communication_GA/key reference/仿真底座证据链/LTE移动网络性能参数实验研究-24-06615-v2.pdf")})
- [论文无人机_仿真证据链_003]({note_link(ev_uav_measure, "D:/UAV_Communication_GA/key reference/仿真底座证据链/On_the_Downlink_SINR_Meta_Distribution_of_UAV-Assisted_Wireless_Networks.pdf")})
- `D:\\UAV_Communication_GA\\12\\output\\measured_corner_compare\\corner_case_summary.json`

## 节点判断

- 这部分把理论模型和公开实测数据接起来了。
- 从结果上看，模型最差区域在 AERPAW 和 Dryad 数据集上都明显低于实测轨迹均值，说明模型具备“找最差通信区”的能力。
""",
    )

    outcome_note(
        ev_sat_papers,
        "证据 卫星 EMC 参考论文",
        f"""
## 节点总结

这组文献是 MATLAB 卫星模型与专利仿真支撑材料共用的理论依据，主题集中在卫星/空间系统 EMC 设计、评估和验证方法。

## 关联原文位置

- [论文卫星_卫星参考论文_001]({note_link(ev_sat_papers, "D:/一汽项目/EMC正向设计写作模板参考/某大型卫星电磁兼容性设计与验证_崔相臣.pdf")})
- [论文卫星_卫星参考论文_002]({note_link(ev_sat_papers, "D:/一汽项目/EMC正向设计写作模板参考/空间站用舱内风机电磁兼容性设计.pdf")})
- [论文卫星_卫星参考论文_003]({note_link(ev_sat_papers, "D:/一汽项目/车载低轨卫星通信系统EMC性能正向设计技术研究方案/论文/基于近场扫描的移动通信卫星EMC性能评估.pdf")})
- [论文卫星_卫星参考论文_004]({note_link(ev_sat_papers, "D:/一汽项目/车载低轨卫星通信系统EMC性能正向设计技术研究方案/论文/车载通信设备EMC评估方法及其应用_孟晓姣.pdf")})

## 节点判断

- 这些文献分别覆盖系统级 EMC 设计、设备级 EMC 设计、近场扫描评估和车载通信设备评估方法。
- 因此它们同时支撑 MATLAB 卫星模型的工程合理性，也支撑专利仿真材料中的“为什么要做这类仿真”。
""",
    )

    outcome_note(
        ev_sat_plan,
        "证据 车载低轨卫星研究方案",
        f"""
## 节点总结

这组材料是卫星成果最直接的项目报告层证据。研究方案、总体设计方案和最终稿明确说明了项目目标：在车载低轨卫星通信系统设计早期开展 EMC 正向设计。

## 关联原文位置

- [正向设计规范_研究方案_001]({note_link(ev_sat_plan, "D:/一汽项目/车载低轨卫星通信系统EMC性能正向设计技术研究方案/【工创中心】低轨卫星通信系统_总体设计方案(1).docx")})
- [正向设计规范_研究方案_002]({note_link(ev_sat_plan, "D:/一汽项目/车载低轨卫星通信系统EMC性能正向设计技术研究方案/最终稿_车载低轨卫星通信系统EMC性能正向设计技术研究方案.docx")})
- [正向设计规范_研究方案_003]({note_link(ev_sat_plan, "D:/一汽项目/车载低轨卫星通信系统EMC性能正向设计技术研究方案/车载低轨卫星通信系统EMC正向设计技术研究方案初版.docx")})

## 节点判断

- 研究方案正文已经明确提出：由于传统事后整改成本高、周期长，因此需要在设计初期系统性地开展 EMC 正向设计。
- 这正好解释了 MATLAB 卫星模型和专利仿真支撑材料的工程用途。
""",
    )

    outcome_note(
        ev_sat_std,
        "证据 卫星 EMC 规范与对照",
        f"""
## 节点总结

这组材料是卫星成果的规范层与专利映射层证据，说明模型和专利支撑工作并不是脱离标准单独存在，而是与企业标准、测量规范和专利条款对照耦合。

## 关联原文位置

- [正向设计规范_标准规范_001]({note_link(ev_sat_std, "D:/一汽项目/车载低轨卫星通信系统EMC性能正向设计技术研究方案/JA 3700-MH-3乘用车电气电子零部件EMC技术条件-11-3-21pdf.pdf")})
- [正向设计规范_标准规范_002]({note_link(ev_sat_std, "D:/一汽项目/车载低轨卫星通信系统EMC性能正向设计技术研究方案/Q_FC_XXX-2025_车载低轨卫星通信系统EMC性能正向设计规范.docx")})
- [正向设计规范_标准规范_003]({note_link(ev_sat_std, "D:/一汽项目/车载低轨卫星通信系统EMC性能正向设计技术研究方案/标准/车载卫星通信设备电磁兼容性要求和测量方法.pdf")})
- [专利卫星_专利标准对照_001]({note_link(ev_sat_std, "D:/一汽项目/车载低轨卫星通信系统EMC性能正向设计技术研究方案/车载低轨卫星通信系统EMC正向设计规范_专利与标准对照文件.docx")})

## 节点判断

- `Q_FC_XXX-2025` 已经是规范化文本，内容至少覆盖范围、引用文件、术语和 EMC 正向设计总体要求。
- “专利与标准对照文件”进一步说明：专利思路、标准条款和工程规范之间已经被显式对应起来。
""",
    )


def write_obsidian_config() -> None:
    cfg_dir = VAULT / ".obsidian"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    write_text(cfg_dir / "app.json", json.dumps({"alwaysUpdateLinks": True, "promptDelete": False}, ensure_ascii=False, indent=2))
    write_text(
        cfg_dir / "core-plugins.json",
        json.dumps(["file-explorer", "search", "backlink", "graph", "outgoing-link"], ensure_ascii=False, indent=2),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="按中文逻辑分类重建 Obsidian 知识库。")
    parser.parse_args()

    reset_vault()
    write_obsidian_config()
    records = collect_records()

    write_text(VAULT / "00_知识库总览.md", build_home(records))
    write_text(CHAT_DIR / "README.md", build_chat_readme())

    for category in CATEGORY_ORDER:
        write_text(category_note_path(category), build_category_note(category, records))
        role_groups: Dict[str, List[KnowledgeRecord]] = defaultdict(list)
        for record in records:
            if record.category == category:
                role_groups[record.role].append(record)
        for role, role_records in sorted(role_groups.items()):
            write_text(role_note_path(category, role), build_role_note(category, role, role_records))

    for record in records:
        write_text(Path(record.note_path), build_card(record))

    build_outcome_graph(records)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "workspace_root": str(ROOT),
        "vault_root": str(VAULT),
        "card_count": len(records),
        "categories": CATEGORY_ORDER,
        "records": [asdict(record) for record in records],
    }
    write_text(AUTO_DIR / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Vault rebuilt at: {VAULT}")
    print(f"Knowledge cards: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
