import argparse
import datetime as _dt
import os
import posixpath
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from xml.sax.saxutils import escape as _xml_escape

from PIL import Image


EMU_PER_INCH = 914400
EMU_PER_PX_96DPI = 9525  # 914400 / 96


SUPPORTED_EXTS = {
    ".png": ("image/png", "png"),
    ".jpg": ("image/jpeg", "jpg"),
    ".jpeg": ("image/jpeg", "jpeg"),
    ".bmp": ("image/bmp", "bmp"),
    ".gif": ("image/gif", "gif"),
    ".webp": ("image/webp", "webp"),
    ".tif": ("image/tiff", "tif"),
    ".tiff": ("image/tiff", "tiff"),
}


@dataclass(frozen=True)
class ImageItem:
    src_path: Path
    media_name: str  # e.g., image1.png
    rel_id: str  # e.g., rId1
    cx_emu: int
    cy_emu: int
    content_type: str


def _iter_images(input_dir: Path, recursive: bool) -> Iterable[Path]:
    if not input_dir.exists():
        return []
    pattern = "**/*" if recursive else "*"
    for path in input_dir.glob(pattern):
        if not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def _dpi_from_pil(img: Image.Image) -> Tuple[float, float]:
    dpi = img.info.get("dpi")
    if isinstance(dpi, tuple) and len(dpi) == 2 and dpi[0] and dpi[1]:
        return float(dpi[0]), float(dpi[1])
    return 96.0, 96.0


def _px_to_emu(px: int, dpi: float) -> int:
    if not dpi:
        return int(px * EMU_PER_PX_96DPI)
    return int(px * (EMU_PER_INCH / dpi))


def _fit_to_width(cx_emu: int, cy_emu: int, max_width_emu: int) -> Tuple[int, int]:
    if cx_emu <= max_width_emu:
        return cx_emu, cy_emu
    scale = max_width_emu / float(cx_emu)
    return int(cx_emu * scale), int(cy_emu * scale)


def _read_image_size_emu(path: Path, max_width_emu: int) -> Tuple[int, int]:
    with Image.open(path) as img:
        w_px, h_px = img.size
        dpi_x, dpi_y = _dpi_from_pil(img)
    cx = _px_to_emu(w_px, dpi_x)
    cy = _px_to_emu(h_px, dpi_y)
    return _fit_to_width(cx, cy, max_width_emu)


def _content_types_xml(image_content_types: Dict[str, str]) -> str:
    defaults = {
        "rels": "application/vnd.openxmlformats-package.relationships+xml",
        "xml": "application/xml",
        **image_content_types,
    }

    default_entries = "\n".join(
        f'  <Default Extension="{_xml_escape(ext)}" ContentType="{_xml_escape(ct)}"/>'
        for ext, ct in sorted(defaults.items())
    )

    overrides = "\n".join(
        [
            '  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>',
            '  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
            '  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
        ]
    )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        f"{default_entries}\n"
        f"{overrides}\n"
        "</Types>\n"
    )


def _package_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>\n'
        '  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>\n'
        '  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>\n'
        "</Relationships>\n"
    )


def _document_rels_xml(items: Sequence[ImageItem]) -> str:
    rels = []
    for it in items:
        rels.append(
            f'  <Relationship Id="{_xml_escape(it.rel_id)}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/{_xml_escape(it.media_name)}"/>'
        )
    rels_xml = "\n".join(rels)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        f"{rels_xml}\n"
        "</Relationships>\n"
    )


def _core_props_xml(title: str) -> str:
    created = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    title_escaped = _xml_escape(title)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
        f"  <dc:title>{title_escaped}</dc:title>\n"
        "  <dc:creator>Codex CLI</dc:creator>\n"
        f'  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>\n'
        f'  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>\n'
        "</cp:coreProperties>\n"
    )


def _app_props_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Properties '
        'xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">\n'
        "  <Application>Codex CLI</Application>\n"
        "</Properties>\n"
    )


def _doc_paragraph_text(text: str) -> str:
    return (
        "<w:p>"
        "<w:r>"
        f"<w:t>{_xml_escape(text)}</w:t>"
        "</w:r>"
        "</w:p>\n"
    )


def _doc_paragraph_page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n'


def _doc_paragraph_image(it: ImageItem, docpr_id: int, name: str) -> str:
    # Minimal inline drawing (WordprocessingML + DrawingML)
    # Ref: ECMA-376; uses wp:inline with a:blip r:embed="{rId}"
    name_escaped = _xml_escape(name)
    return (
        "<w:p><w:r><w:drawing>"
        '<wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{it.cx_emu}" cy="{it.cy_emu}"/>'
        f'<wp:docPr id="{docpr_id}" name="{name_escaped}"/>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        "<pic:nvPicPr>"
        f'<pic:cNvPr id="{docpr_id}" name="{name_escaped}"/>'
        "<pic:cNvPicPr/>"
        "</pic:nvPicPr>"
        "<pic:blipFill>"
        f'<a:blip xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:embed="{_xml_escape(it.rel_id)}"/>'
        "<a:stretch><a:fillRect/></a:stretch>"
        "</pic:blipFill>"
        "<pic:spPr>"
        "<a:xfrm>"
        f'<a:off x="0" y="0"/>'
        f'<a:ext cx="{it.cx_emu}" cy="{it.cy_emu}"/>'
        "</a:xfrm>"
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        "</pic:spPr>"
        "</pic:pic>"
        "</a:graphicData>"
        "</a:graphic>"
        "</wp:inline>"
        "</w:drawing></w:r></w:p>\n"
    )


def _document_xml(title: str, items: Sequence[ImageItem]) -> str:
    body = []
    body.append(_doc_paragraph_text(title))
    body.append(_doc_paragraph_page_break())

    docpr_id = 1
    for idx, it in enumerate(items, start=1):
        filename = it.src_path.name
        body.append(_doc_paragraph_text(f"{idx}. {filename}"))
        body.append(_doc_paragraph_image(it, docpr_id=docpr_id, name=filename))
        docpr_id += 1
        if idx != len(items):
            body.append(_doc_paragraph_page_break())

    sect_pr = (
        "<w:sectPr>"
        '<w:pgSz w="11906" h="16838"/>'  # A4 in twips
        '<w:pgMar top="1440" right="1440" bottom="1440" left="1440" header="720" footer="720" gutter="0"/>'
        "</w:sectPr>"
    )

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n'
        "<w:body>\n"
        + "".join(body)
        + sect_pr
        + "\n</w:body>\n</w:document>\n"
    )


def build_docx(
    images: Sequence[Path],
    output_path: Path,
    title: str,
    max_width_in: float,
) -> Path:
    max_width_emu = int(max_width_in * EMU_PER_INCH)

    image_content_types: Dict[str, str] = {}
    items: List[ImageItem] = []
    for idx, img_path in enumerate(images, start=1):
        ext = img_path.suffix.lower()
        content_type, ext_no_dot = SUPPORTED_EXTS[ext]
        image_content_types[ext_no_dot] = content_type
        media_name = f"image{idx}{ext}"
        rel_id = f"rId{idx}"
        cx_emu, cy_emu = _read_image_size_emu(img_path, max_width_emu=max_width_emu)
        items.append(
            ImageItem(
                src_path=img_path,
                media_name=media_name,
                rel_id=rel_id,
                cx_emu=cx_emu,
                cy_emu=cy_emu,
                content_type=content_type,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _content_types_xml(image_content_types))
        zf.writestr("_rels/.rels", _package_rels_xml())
        zf.writestr("docProps/core.xml", _core_props_xml(title))
        zf.writestr("docProps/app.xml", _app_props_xml())
        zf.writestr("word/document.xml", _document_xml(title, items))
        zf.writestr("word/_rels/document.xml.rels", _document_rels_xml(items))

        for it in items:
            arcname = posixpath.join("word", "media", it.media_name)
            zf.write(it.src_path, arcname=arcname)

    return output_path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a .docx that embeds images (no python-docx dependency)."
    )
    p.add_argument(
        "input",
        nargs="?",
        default="input_images",
        help="Input directory containing images (default: input_images)",
    )
    p.add_argument(
        "output",
        nargs="?",
        default="outline_images.docx",
        help="Output .docx path (default: outline_images.docx)",
    )
    p.add_argument(
        "--title",
        default="Images",
        help="Document title (default: Images)",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input directory recursively",
    )
    p.add_argument(
        "--max-width-in",
        type=float,
        default=6.0,
        help="Max image width in inches (default: 6.0)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    input_dir = Path(args.input)
    output_path = Path(args.output)

    images = sorted(_iter_images(input_dir, recursive=args.recursive), key=lambda p: str(p))
    if not images:
        exts = ", ".join(sorted({e for e in SUPPORTED_EXTS.keys()}))
        raise SystemExit(
            f"No images found under '{input_dir}'. Put your screenshots there (supported: {exts})."
        )

    build_docx(
        images=images,
        output_path=output_path,
        title=args.title,
        max_width_in=float(args.max_width_in),
    )
    print(f"Wrote: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

