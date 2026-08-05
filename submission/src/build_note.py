"""
build_note -- подстановка посчитанных чисел в шаблон Research Note и рендер PDF.

Зачем так. Текст отчёта и таблицы читают один и тот же note_values.json,
поэтому утверждения в прозе физически не могут разойтись с числами
(в первой части проекта расхождение текста и таблиц ловилось трижды).

Запуск:
    python build_note.py --values note_values.json --template research_note.md \
                         --out Research_Note.pdf --figdir logs

Если note_values.json нет, шаблон рендерится с прочерками на месте чисел --
получается корректно свёрстанный документ, в который остаётся подставить
результаты прогона.

Зависимости: reportlab (pip install reportlab). Кириллический шрифт берётся
из поставки matplotlib (DejaVu), то есть ставить ничего дополнительно не надо.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

MISSING = "—"

# Заголовки таблиц: reportlab переносит строку только по пробелам, поэтому
# имена с подчёркиваниями не помещаются в колонку и рвутся посередине слова.
# Замена делается на стороне рендера, чтобы csv сохраняли машинные имена.
HEADERS = {
    "AUC_R@5": "AUC R 5°", "AUC_R@10": "AUC R 10°", "AUC_R@20": "AUC R 20°",
    "AUC_t@5": "AUC t 5°", "AUC_t@10": "AUC t 10°", "AUC_t@20": "AUC t 20°",
    "AUC@5": "AUC 5°", "AUC@10": "AUC 10°", "AUC@20": "AUC 20°",
    "медиана_err_R": "мед. e R, °", "медиана_err_t": "мед. e t, °",
    "отказы_%": "отказы, %", "отказы_pct": "отказы, %",
    "база_м": "база, м", "база_к_глубине": "база / глубина",
    "мс_пара": "мс на пару", "мс_фронтенд": "мс фронтенд",
    "порог_px": "порог, px", "разрешение": "разрешение, px", "stride": "шаг",
}


# ------------------------------------------------------------------- шрифты
def register_fonts() -> tuple[str, str, str]:
    """Ищет DejaVu (кириллица) в системе и в поставке matplotlib."""
    cands = []
    try:
        import matplotlib
        mpl = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        cands.append(mpl)
    except Exception:
        pass
    cands += [Path("/usr/share/fonts/truetype/dejavu"),
              Path("/usr/share/fonts/TTF"),
              Path("C:/Windows/Fonts")]

    def find(name):
        for d in cands:
            p = d / name
            if p.exists():
                return p
        return None

    reg, bold, mono = find("DejaVuSans.ttf"), find("DejaVuSans-Bold.ttf"), find("DejaVuSansMono.ttf")
    if reg is None:
        print("[build_note] DejaVu не найден, кириллица может не отрисоваться")
        return "Helvetica", "Helvetica-Bold", "Courier"
    pdfmetrics.registerFont(TTFont("DejaVu", str(reg)))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(bold or reg)))
    pdfmetrics.registerFont(TTFont("DejaVu-Mono", str(mono or reg)))
    return "DejaVu", "DejaVu-Bold", "DejaVu-Mono"


# -------------------------------------------------------------- подстановка
def substitute(text: str, values: dict) -> tuple[str, list[str]]:
    """Меняет {{ТОКЕН}} на значение; неизвестные токены -> прочерк."""
    missing = []

    def rep(m):
        key = m.group(1).strip()
        if key in values:
            v = values[key]
            return str(v)
        missing.append(key)
        return MISSING

    return re.sub(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}", rep, text), sorted(set(missing))


def inline(s: str) -> str:
    """Минимальная inline-разметка markdown -> разметка reportlab."""
    s = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`(.+?)`", r'<font face="DejaVu-Mono" size="8.5">\1</font>', s)
    s = re.sub(r"\[(.+?)\]\((.+?)\)", r'<link href="\2" color="#1a4f8a">\1</link>', s)
    return s


# ------------------------------------------------------------------- рендер
def build(md_text: str, out_pdf: Path, figdir: Path) -> None:
    reg, bold, mono = register_fonts()
    ss = getSampleStyleSheet()

    body = ParagraphStyle("body", parent=ss["Normal"], fontName=reg, fontSize=9.5,
                          leading=13.5, alignment=TA_JUSTIFY, spaceAfter=5)
    h1 = ParagraphStyle("h1", parent=body, fontName=bold, fontSize=15, leading=19,
                        spaceBefore=14, spaceAfter=8, alignment=0)
    h2 = ParagraphStyle("h2", parent=body, fontName=bold, fontSize=12, leading=16,
                        spaceBefore=11, spaceAfter=5, alignment=0)
    h3 = ParagraphStyle("h3", parent=body, fontName=bold, fontSize=10.5, leading=14,
                        spaceBefore=8, spaceAfter=4, alignment=0)
    title = ParagraphStyle("title", parent=body, fontName=bold, fontSize=17, leading=22,
                           alignment=TA_CENTER, spaceAfter=4)
    sub = ParagraphStyle("sub", parent=body, fontName=reg, fontSize=9.5, leading=13,
                         alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
                         spaceAfter=12)
    formula = ParagraphStyle("formula", parent=body, fontName=mono, fontSize=8.5,
                             leading=12, alignment=TA_CENTER,
                             backColor=colors.HexColor("#f4f4f2"),
                             borderPadding=6, spaceBefore=6, spaceAfter=8)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=3,
                            spaceAfter=2)
    author = ParagraphStyle("author", parent=body, fontSize=11, leading=15,
                            alignment=TA_CENTER, spaceAfter=2)
    affil = ParagraphStyle("affil", parent=body, fontSize=9.5, leading=13,
                           alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
                           spaceAfter=10)
    caption = ParagraphStyle("caption", parent=body, fontSize=8.5, leading=11,
                             alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
                             spaceAfter=8)

    story = []
    lines = md_text.split("\n")
    i, first_h1 = 0, True
    # Абзацы между заголовком и первой цитатой -- титульный блок (автор,
    # принадлежность): центрируются, а не верстаются как основной текст.
    frontmatter = 0

    def flush_table(rows):
        if not rows:
            return
        header, data = [HEADERS.get(c, c) for c in rows[0]], rows[1:]
        cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10.5, alignment=0)
        cell_h = ParagraphStyle("cell_h", parent=cell, fontName=bold)
        tbl = [[Paragraph(inline(c), cell_h) for c in header]] + \
              [[Paragraph(inline(c), cell) for c in r] for r in data]

        # Ширина колонки пропорциональна самому длинному слову в ней, а не
        # делится поровну: иначе узкая колонка с длинным заголовком рвёт
        # слово посередине, а соседняя простаивает пустой.
        total = A4[0] - 36 * mm
        weights = []
        for j in range(len(header)):
            col = [header[j]] + [r[j] if j < len(r) else "" for r in data]
            longest_word = max((len(w) for c in col for w in str(c).split()), default=1)
            mean_len = sum(len(str(c)) for c in col) / len(col)
            weights.append(max(longest_word + 1, mean_len * 0.8, 4))
        # Одна колонка с длинным значением (например именем метода) не должна
        # съедать ширину у остальных: её вес ограничивается сверху. Длинное
        # значение перенесётся по строкам, что лучше, чем разрыв заголовков
        # в соседних колонках посередине слова.
        cap = 2.6 * (sum(weights) / len(weights))
        weights = [min(w, cap) for w in weights]
        ssum = sum(weights)
        widths = [total * w / ssum for w in weights]

        t = Table(tbl, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eaed")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b0b0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#fafafa")]),
        ]))
        story.append(Spacer(1, 4))
        story.append(t)
        story.append(Spacer(1, 8))

    while i < len(lines):
        ln = lines[i].rstrip()

        if not ln.strip():
            i += 1
            continue

        # формулы и блоки кода
        if ln.strip().startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i].rstrip())
                i += 1
            i += 1
            # Порядок важен: сначала экранируем сам амперсанд, потом вставляем
            # неразрывные пробелы. В обратном порядке &nbsp; превращается
            # в &amp;nbsp; и печатается как текст.
            def esc(x):
                x = x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                return x.replace(" ", "&nbsp;")

            txt = "<br/>".join(esc(x) for x in buf)
            story.append(Paragraph(txt, formula))
            continue

        # таблицы
        if ln.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") and c for c in cells):
                    rows.append(cells)
                i += 1
            flush_table(rows)
            continue

        # картинки
        m = re.match(r"!\[(.*?)\]\((.+?)\)", ln.strip())
        if m:
            alt, src = m.group(1), m.group(2)
            p = Path(src)
            if not p.is_absolute():
                p = figdir / src
            if p.exists():
                from reportlab.lib.utils import ImageReader
                iw, ih = ImageReader(str(p)).getSize()
                maxw = A4[0] - 36 * mm
                w = min(maxw, iw * 0.55)
                story.append(KeepTogether([Image(str(p), width=w, height=ih * w / iw),
                                           Paragraph(inline(alt), caption)]))
            else:
                story.append(Paragraph(f"<i>[рисунок не найден: {src}]</i>", caption))
            i += 1
            continue

        if ln.startswith("---"):
            story.append(Spacer(1, 6))
            i += 1
            continue

        if ln.startswith("#"):
            level = len(ln) - len(ln.lstrip("#"))
            txt = inline(ln.lstrip("# ").strip())
            if level == 1 and first_h1:
                story.append(Paragraph(txt, title))
                first_h1 = False
                frontmatter = 2
            else:
                story.append(Paragraph(txt, {1: h1, 2: h1, 3: h2}.get(level, h3)))
            i += 1
            continue

        if ln.startswith("> "):
            frontmatter = 0
            story.append(Paragraph(inline(ln[2:]), sub))
            i += 1
            continue

        if re.match(r"^\s*[-*] ", ln) or re.match(r"^\s*\d+\. ", ln):
            buf = []
            while i < len(lines) and (re.match(r"^\s*[-*] ", lines[i])
                                      or re.match(r"^\s*\d+\. ", lines[i])
                                      or (lines[i].startswith("  ") and lines[i].strip())):
                s = lines[i].strip()
                if re.match(r"^[-*] ", s) or re.match(r"^\d+\. ", s):
                    buf.append(re.sub(r"^([-*]|\d+\.)\s+", "", s))
                elif buf:
                    buf[-1] += " " + s
                i += 1
            for b in buf:
                story.append(Paragraph(inline(b), bullet, bulletText="•"))
            story.append(Spacer(1, 4))
            continue

        # обычный абзац
        buf = [ln]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^\s*(#|\||!\[|```|[-*] |\d+\. |> |---)", lines[i]):
            buf.append(lines[i].rstrip())
            i += 1
        para = " ".join(buf)
        if frontmatter > 0:
            story.append(Paragraph(inline(para), author if frontmatter == 2 else affil))
            frontmatter -= 1
        else:
            story.append(Paragraph(inline(para), sub if para.startswith("_") else body))

    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Research Note: Visual Odometry на ALTO",
                            author="Максим Тишин",
                            subject="Тестовое задание AI / ML Engineer")

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont(reg, 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawCentredString(A4[0] / 2, 9 * mm, str(doc_.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="research_note.md")
    ap.add_argument("--values", default="note_values.json")
    ap.add_argument("--out", default="Research_Note.pdf")
    ap.add_argument("--figdir", default="logs")
    a = ap.parse_args()

    md = Path(a.template).read_text(encoding="utf-8")
    values = {}
    vp = Path(a.values)
    if vp.exists():
        values = json.loads(vp.read_text(encoding="utf-8"))
    else:
        print(f"[build_note] {vp} не найден -- числа будут заменены прочерками")

    md, missing = substitute(md, values)
    if missing:
        print("[build_note] не подставлены токены:", ", ".join(missing))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    build(md, Path(a.out), Path(a.figdir))
    print("[build_note] готово:", a.out)


if __name__ == "__main__":
    main()
