from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VAULT = ROOT / "obsidian_vault"
CONVERSATIONS = VAULT / "30_对话沉淀"


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\-. \u4e00-\u9fff]+", "", text)
    text = re.sub(r"\s+", "-", text)
    return text[:80] or "note"


def read_content(args: argparse.Namespace) -> str:
    if args.content_file:
        return Path(args.content_file).read_text(encoding="utf-8")
    if args.stdin:
        return sys.stdin.read()
    if args.content:
        return args.content
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a Codex output into the Obsidian vault.")
    parser.add_argument("--title", required=True, help="Note title")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--folder", default="", help="Optional extra folder under 30_Conversation_Notes")
    parser.add_argument("--content", default="", help="Inline note body")
    parser.add_argument("--content_file", default="", help="Read note body from a file")
    parser.add_argument("--stdin", action="store_true", help="Read note body from stdin")
    args = parser.parse_args()

    ts = datetime.now()
    day_folder = CONVERSATIONS / ts.strftime("%Y-%m-%d")
    if args.folder:
        day_folder = day_folder / slugify(args.folder)
    day_folder.mkdir(parents=True, exist_ok=True)

    note_path = day_folder / f"{ts.strftime('%H%M%S')}_{slugify(args.title)}.md"
    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    body = read_content(args).strip()

    text = [
        "---",
        f'title: "{args.title}"',
        f'created: "{ts.strftime("%Y-%m-%d %H:%M:%S")}"',
        f"tags: {tags!r}",
        "---",
        f"# {args.title}",
        "",
    ]
    if body:
        text.extend([body, ""])
    else:
        text.extend(["No content supplied.", ""])

    note_path.write_text("\n".join(text), encoding="utf-8")
    print(note_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
