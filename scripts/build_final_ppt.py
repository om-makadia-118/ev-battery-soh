"""Build the finalized BMSCE Project Work - 2 PowerPoint strictly matching the attached template.

Directly loads the attached BMSCE starter PPTX (`media_1795782314121_2981240e.pptx`),
updates Slide 1 (Title), Slide 2 (Agenda), Slide 4 (Suggestions / Questions), and Slide 5
(Thank you!) IN-PLACE preserving exact template styling/fonts/colors, and inserts pure
white, classic Times New Roman academic content slides between Agenda and Q&A.

Output: reports/EV_Battery_SOH_Final_Presentation.pptx
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.xmlchemy import OxmlElement

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = Path(
    "/Users/divyamjain/.gemini/antigravity/brain/"
    "d7a8b650-36ed-4f79-aa02-03da72c12a38/.user_uploaded/"
    "media_1795782314121_2981240e.pptx"
)
ASSETS = ROOT / "reports" / "ppt_assets"
RESULTS = ROOT / "results" / "full_study"
OUT_PPTX = ROOT / "reports" / "EV_Battery_SOH_Final_Presentation.pptx"

# Exact Template Colors & Fonts
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x00, 0x00, 0x00)
DARK_TEXT = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x55, 0x55, 0x55)
BMS_RED = RGBColor(0xC0, 0x00, 0x00)      # Template "DEPARTMENT OF MACHINE LEARNING" & "Agenda"
BMS_BLUE = RGBColor(0x00, 0x48, 0x7E)     # Template "Course: Project Work - 2" & "Project Title"
BMS_PURPLE = RGBColor(0x70, 0x30, 0xA0)   # Template "SEMESTER END EXAMINATION"
TEAL_OK = RGBColor(0x00, 0x66, 0x44)      # Subtle dark emerald for best scores
RULE_LIGHT = RGBColor(0xD9, 0xE2, 0xEC)
HEADER_TINT = RGBColor(0xF4, 0xF7, 0xFA)  # Ultra-faint academic table header wash

FONT_TNR = "Times New Roman"


def set_solid_bg(slide, color: RGBColor = WHITE) -> None:
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_hline(slide, x: float, y: float, w: float, color: RGBColor = RULE_LIGHT, thickness_pt: float = 0.75):
    sh = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Pt(thickness_pt)
    )
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def add_textbox(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str = "",
    font_size: float = 14,
    bold: bool = False,
    italic: bool = False,
    color: RGBColor = BLACK,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    font_name: str = FONT_TNR,
):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    if text:
        p = tf.paragraphs[0]
        p.alignment = align
        r = p.add_run()
        r.text = text
        r.font.name = font_name
        r.font.size = Pt(font_size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
    return tb


def set_p_bullet(p, bullet_sz_pt: float = 14) -> None:
    """Apply classic template round bullet (<a:buChar char="•"/>) via OOXML."""
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(0.28))))
    pPr.set("indent", str(-int(Inches(0.20))))
    for child in list(pPr):
        tag = child.tag.split("}")[-1]
        if tag in ("buNone", "buChar", "buFont", "buSzPts", "buClr"):
            pPr.remove(child)
    buClr = OxmlElement("a:buClr")
    srgb = OxmlElement("a:srgbClr")
    srgb.set("val", "000000")
    buClr.append(srgb)
    buSz = OxmlElement("a:buSzPts")
    buSz.set("val", str(int(bullet_sz_pt * 100)))
    buFont = OxmlElement("a:buFont")
    buFont.set("typeface", "Arial")
    buChar = OxmlElement("a:buChar")
    buChar.set("char", "•")
    pPr.append(buClr)
    pPr.append(buSz)
    pPr.append(buFont)
    pPr.append(buChar)


def add_bullets(
    tf,
    items: list[tuple[str, str]],
    font_size: float = 14.0,
    space_after: float = 8.0,
    line_spacing: float = 1.18,
    lead_color: RGBColor = BLACK,
    text_color: RGBColor = BLACK,
    first_para_empty: bool = True,
) -> None:
    for idx, (lead, body) in enumerate(items):
        p = tf.paragraphs[0] if (idx == 0 and first_para_empty) else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        set_p_bullet(p, bullet_sz_pt=font_size)
        if lead:
            rl = p.add_run()
            rl.text = lead + " "
            rl.font.name = FONT_TNR
            rl.font.size = Pt(font_size)
            rl.font.bold = True
            rl.font.color.rgb = lead_color
        if body:
            rt = p.add_run()
            rt.text = body
            rt.font.name = FONT_TNR
            rt.font.size = Pt(font_size)
            rt.font.color.rgb = text_color


def add_fit_picture(slide, img_path: Path | str, bx: float, by: float, bw: float, bh: float):
    img_path = Path(img_path)
    with Image.open(img_path) as im:
        pw, ph = im.size
    ratio = pw / ph
    box_ratio = bw / bh
    if ratio >= box_ratio:
        fw = bw
        fh = fw / ratio
    else:
        fh = bh
        fw = fh * ratio
    fx = bx + (bw - fw) / 2.0
    fy = by + (bh - fh) / 2.0
    return slide.shapes.add_picture(str(img_path), Inches(fx), Inches(fy), Inches(fw), Inches(fh))


def add_template_header(slide, title: str, subtitle: str = "", page_num: int | None = None) -> None:
    """Match Template Slide 2 ('Agenda' centered #C00000 Times New Roman) on pure white."""
    set_solid_bg(slide, WHITE)
    # Remove unused body placeholder if inherited from OBJECT layout (only using BLANK, so none!)
    add_textbox(
        slide, 0.50, 0.22, 12.33, 0.62, title,
        font_size=30.0, bold=False, color=BMS_RED, align=PP_ALIGN.CENTER, font_name=FONT_TNR
    )
    if subtitle:
        add_textbox(
            slide, 0.80, 0.84, 11.73, 0.30, subtitle,
            font_size=13.0, italic=True, color=BMS_BLUE, align=PP_ALIGN.CENTER, font_name=FONT_TNR
        )
        add_hline(slide, 0.90, 1.18, 11.53, color=RULE_LIGHT, thickness_pt=0.75)
    else:
        add_hline(slide, 0.90, 0.92, 11.53, color=RULE_LIGHT, thickness_pt=0.75)

    if page_num is not None:
        add_textbox(
            slide, 11.80, 7.06, 0.95, 0.28, str(page_num),
            font_size=11.0, bold=False, color=MUTED, align=PP_ALIGN.RIGHT, font_name=FONT_TNR
        )


def set_booktabs_borders(cell, top_w: int = 0, bot_w: int = 6350, color_hex: str = "CBD5E1") -> None:
    """Clean horizontal academic booktabs borders (zero vertical grid bars)."""
    tcPr = cell._tc.get_or_add_tcPr()
    for edge in ("lnL", "lnR", "lnT", "lnB"):
        for old in tcPr.findall(f"{{http://schemas.openxmlformats.org/drawingml/2006/main}}{edge}"):
            tcPr.remove(old)
        ln = OxmlElement(f"a:{edge}")
        ln.set("cap", "flat")
        ln.set("cmpd", "sng")
        ln.set("algn", "ctr")
        w = top_w if edge == "lnT" else (bot_w if edge == "lnB" else 0)
        if w == 0:
            ln.set("w", "0")
            ln.append(OxmlElement("a:noFill"))
        else:
            ln.set("w", str(w))
            sf = OxmlElement("a:solidFill")
            sr = OxmlElement("a:srgbClr")
            sr.set("val", color_hex)
            sf.append(sr)
            ln.append(sf)
        tcPr.append(ln)


def add_booktabs_table(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    col_widths: list[float],
    headers: list[str],
    rows: list[list[str]],
    font_size: float = 11.5,
    header_size: float = 12.0,
    col_aligns: list | None = None,
    bold_last_row: bool = False,
    bold_cells: set[tuple[int, int]] | None = None,
    warn_cells: set[tuple[int, int]] | None = None,
):
    n_rows = 1 + len(rows)
    n_cols = len(headers)
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = shape.table
    # Remove default PowerPoint table style tinting so pure white academic booktabs shines
    tblPr = tbl._tbl.tblPr
    for attr in ("firstRow", "bandRow"):
        tblPr.set(attr, "0")

    tot = sum(col_widths)
    for idx, cw in enumerate(col_widths):
        tbl.columns[idx].width = Inches(w * (cw / tot))

    for r_idx in range(n_rows):
        is_hdr = r_idx == 0
        is_last = r_idx == n_rows - 1
        row_vals = headers if is_hdr else rows[r_idx - 1]
        for c_idx in range(n_cols):
            cell = tbl.cell(r_idx, c_idx)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(0.07)
            cell.margin_right = Inches(0.07)
            cell.margin_top = Inches(0.04)
            cell.margin_bottom = Inches(0.04)
            cell.fill.solid()
            if is_hdr:
                cell.fill.fore_color.rgb = HEADER_TINT
                set_booktabs_borders(cell, top_w=19050, bot_w=19050, color_hex="00487E")
            elif is_last:
                cell.fill.fore_color.rgb = HEADER_TINT if bold_last_row else WHITE
                top_sep = 12700 if bold_last_row else 0
                set_booktabs_borders(cell, top_w=top_sep, bot_w=19050, color_hex="00487E")
            else:
                cell.fill.fore_color.rgb = WHITE
                set_booktabs_borders(cell, top_w=0, bot_w=6350, color_hex="D8E0E8")

            p = cell.text_frame.paragraphs[0]
            p.alignment = col_aligns[c_idx] if col_aligns else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = row_vals[c_idx]
            r.font.name = FONT_TNR
            r.font.size = Pt(header_size if is_hdr else font_size)
            is_bold = is_hdr or (bold_last_row and is_last) or (bold_cells and (r_idx - 1, c_idx) in bold_cells)
            r.font.bold = bool(is_bold)
            if is_hdr:
                r.font.color.rgb = BMS_BLUE
            elif warn_cells and (r_idx - 1, c_idx) in warn_cells:
                r.font.color.rgb = BMS_RED
                r.font.bold = True
            elif bold_cells and (r_idx - 1, c_idx) in bold_cells:
                r.font.color.rgb = BMS_BLUE
            else:
                r.font.color.rgb = BLACK
    return shape


def add_subheading(slide, x: float, y: float, w: float, text: str, size: float = 17.0):
    add_textbox(slide, x, y, w, 0.34, text, font_size=size, bold=True, color=BMS_BLUE, font_name=FONT_TNR)


def edit_starter_slide1(slide1) -> None:
    """Edit Slide 1 in-place, matching every font, position, and style of the BMSCE template."""
    # Shape 2: Course / Code / Project Title (#00487E Bold Times New Roman)
    sh2 = slide1.shapes[2]
    sh2.top = Inches(2.55)
    sh2.left = Inches(0.65)
    sh2.width = Inches(12.03)
    sh2.height = Inches(1.80)
    tf2 = sh2.text_frame
    tf2.word_wrap = True
    # P0 = Course: Project Work - 2 (keep), P1 = Course Code (keep), P2 = Project Title (replace text)
    p_title = tf2.paragraphs[2]
    p_title.space_before = Pt(4)
    p_title.line_spacing = 1.15
    for run in p_title.runs:
        run.text = (
            "Cross-Dataset Generalization and Few-Shot Adaptation of\n"
            "Machine-Learning Models for Electric Vehicle Battery State-of-Health Estimation"
        )
        run.font.size = Pt(22)
        run.font.bold = True
        run.font.name = FONT_TNR
        run.font.color.rgb = BMS_BLUE

    # Shape 5: SEMESTER END EXAMINATION (keep text/purple style, slight y safety check)
    sh5 = slide1.shapes[5]
    sh5.top = Inches(4.52)

    # Shape 3: Guide block (keep exact template position/look; fill Dr. M. Dakshayini)
    sh3 = slide1.shapes[3]
    sh3.left = Inches(9.20)
    sh3.top = Inches(5.40)
    sh3.width = Inches(3.60)
    sh3.height = Inches(1.40)
    tf3 = sh3.text_frame
    guide_lines = [
        ("Under the Guidance of", 12, True),
        ("Dr. M. Dakshayini", 14, True),
        ("Head of Department", 12, False),
        ("Department of Machine Learning", 12, False),
        ("B.M.S. College of Engineering", 12, False),
    ]
    # Clear existing runs & repopulate cleanly using template's exact Times New Roman Black
    for _ in range(len(tf3.paragraphs) - 1):
        p_el = tf3.paragraphs[-1]._p
        p_el.getparent().remove(p_el)
    for idx, (txt, sz, is_b) in enumerate(guide_lines):
        p = tf3.paragraphs[0] if idx == 0 else tf3.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(2)
        if idx == 0:
            p.clear()
        r = p.add_run()
        r.text = txt
        r.font.name = FONT_TNR
        r.font.size = Pt(sz)
        r.font.bold = is_b
        r.font.color.rgb = BLACK

    # Shape 4: Presented By block (plain Times New Roman Name + tab + USN matching template!)
    sh4 = slide1.shapes[4]
    sh4.left = Inches(1.071)
    sh4.top = Inches(5.12)
    sh4.width = Inches(5.20)
    sh4.height = Inches(2.20)
    tf4 = sh4.text_frame
    tf4.word_wrap = True
    for _ in range(len(tf4.paragraphs) - 1):
        p_el = tf4.paragraphs[-1]._p
        p_el.getparent().remove(p_el)

    student_lines = [
        ("Presented By,", 12, True, False),
        ("Divyam Jain\t1BM23AI063", 14, False, True),
        ("D. Pranav Sai\t1BM23AI064", 14, False, True),
        ("Goutham T. G.\t1BM23AI070", 14, False, True),
        ("Om Manish Makadia\t1BM23AI118", 14, False, True),
        ("", 6, False, False),
        ("Semester & Section: 7 – ‘A, B’", 12, False, False),
        ("Date: November 2026", 14, False, False),
    ]
    for idx, (txt, sz, is_b, has_tab) in enumerate(student_lines):
        p = tf4.paragraphs[0] if idx == 0 else tf4.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(1)
        if idx == 0:
            p.clear()
        if has_tab:
            pPr = p._p.get_or_add_pPr()
            for old in list(pPr):
                if old.tag.endswith("tabLst"):
                    pPr.remove(old)
            tabLst = OxmlElement("a:tabLst")
            tab = OxmlElement("a:tab")
            tab.set("pos", str(int(Inches(2.85))))  # Clean USN column alignment
            tab.set("algn", "l")
            tabLst.append(tab)
            pPr.append(tabLst)
        if txt:
            r = p.add_run()
            r.text = txt
            r.font.name = FONT_TNR
            r.font.size = Pt(sz)
            r.font.bold = is_b
            r.font.color.rgb = BLACK


def edit_starter_agenda_slide2(slide2) -> None:
    """Keep Slide 2 ('Agenda') completely true to the template — clean bullet casing fix only."""
    sh2 = slide2.shapes[2]
    # Keep exact coordinates/font/bullets of template Slide 2; update wording cleanly only where minor grammar
    tf = sh2.text_frame
    replacements = {
        "Literature Survey and Research Gap’s ": "Literature Survey and Research Gaps",
        "Problem statement.": "Problem Statement",
        "Detailed design": "Detailed Design",
        "Conclusion\xa0": "Conclusion",
        "Paper published": "Paper Published / Submission Status",
    }
    for p in tf.paragraphs:
        for r in p.runs:
            if r.text in replacements:
                r.text = replacements[r.text]


def build_deck() -> Path:
    prs = Presentation(str(TEMPLATE_PATH))
    blank = prs.slide_layouts[6]

    # Starter slides in template:
    # 0 = Title, 1 = Agenda, 2 = Blank OBJECT slide (reuse as Slide 3 Introduction!),
    # 3 = Suggestions/Questions, 4 = Thank you !
    edit_starter_slide1(prs.slides[0])
    edit_starter_agenda_slide2(prs.slides[1])

    sldIdLst = prs.slides._sldIdLst
    qa_sldId = sldIdLst[3]
    ty_sldId = sldIdLst[4]

    # Reuse Starter Slide 3 (clear its empty placeholder shape & switch to BLANK clean state)
    s3_reuse = prs.slides[2]
    for sh in list(s3_reuse.shapes):
        sp = sh._element
        sp.getparent().remove(sp)

    content_slides = []

    def new_content_slide():
        sl = prs.slides.add_slide(blank)
        content_slides.append(sl)
        return sl

    # =========================================================================
    # SLIDE 3: 01 INTRODUCTION — CONTEXT & DATA HETEROGENEITY
    # =========================================================================
    s3 = s3_reuse
    add_template_header(
        s3, "Introduction",
        "Project Scope & Multi-Repository Heterogeneity (190 Cells, 132,389 Observations)", page_num=3
    )
    add_subheading(s3, 0.70, 1.38, 4.40, "Project Context & Scope")
    tb3 = add_textbox(s3, 0.70, 1.80, 4.45, 4.95)
    add_bullets(tb3.text_frame, [
        ("State of Health (SOH):", "Ratio of present capacity to rated capacity (SOH = 100 × Qt / Qrated), key to EV range, safety & second-life reuse."),
        ("Transfer Gap:", "Models validated on one laboratory batch often fail when chemistry, temperature, protocol & cycle range change."),
        ("Five Public Labs:", "Harmonized NASA, CALCE, XJTU, Oxford & MATR into 190 physical cells & 132,389 observations."),
        ("Single Shared Predictor:", "Because V/I/T sensors are missing in ≥1 repository, cycle index (ct) is the sole predictor shared across all 5 sources."),
        ("Benchmark Scale:", "6,120 executed experiments + 320 logged infeasible runs across 5 random seeds (11, 22, 33, 44, 55)."),
    ], font_size=13.5, space_after=9)

    add_fit_picture(s3, ASSETS / "fig1_heterogeneity.png", 5.30, 1.44, 7.35, 4.35)
    add_textbox(
        s3, 5.35, 5.92, 7.25, 0.80,
        "Fig. 1: (a) SOH trajectories across all 190 cells showing the 49× cycle-span shift (NASA ≤ 168 vs. Oxford ≤ 8,200) & CALCE low-SOH noise; (b) Observation density dominated by MATR (86.6% of all rows).",
        font_size=11.5, italic=True, color=MUTED, align=PP_ALIGN.CENTER
    )

    # =========================================================================
    # SLIDE 4: 01 INTRODUCTION — INVENTORY & SCOPE BOUNDARY
    # =========================================================================
    s4 = new_content_slide()
    add_template_header(
        s4, "Introduction — Dataset Inventory & Scope",
        "Harmonized Five-Repository Profile & Evidence Boundary (Tables 1.1 & 1.2)", page_num=4
    )
    add_booktabs_table(
        s4, 0.70, 1.36, 11.93, 3.25,
        [1.1, 1.65, 0.85, 0.75, 0.70, 1.05, 0.95, 1.15, 3.73],
        ["Dataset", "Chemistry / Form", "Qrated", "Temp", "Cells", "Obs.", "Max Cyc.", "EOL (<80%)", "Conditions & Verified Caveats"],
        [
            ["NASA", "LCO / 18650", "2.0 Ah", "24°C", "4", "636", "168", "4 (100%)", "4 local source files (B0005–07,18); original EOL 70%"],
            ["CALCE CS2", "LCO / Prismatic", "1.1 Ah", "RT", "15", "9,608", "1,308", "15 (100%)", "13 Arbin + 2 CADEX; 12.5% noisy obs. (<50% SOH)"],
            ["XJTU", "NCM / 18650", "2.0 Ah", "RT", "23", "6,938", "419", "4 (17%)", "2C & 3C fast-charge batches of 55-cell release"],
            ["Oxford", "LCO / Pouch", "0.74 Ah", "40°C", "8", "519", "8,200", "8 (100%)", "Artemis dynamic drive; sparse ~100-cycle checkups"],
            ["MATR", "LFP / 18650", "1.1 Ah", "30°C", "140", "114,688", "2,237", "43 (31%)", "3 batches (86.6% of rows); 140 parsed vs. 124 study"],
            ["Total", "5 Public Labs", "—", "—", "190", "132,389", "8,200", "74 (39%)", "Cycle index shared predictor; no synthetic labels"],
        ],
        font_size=11.0, header_size=11.5,
        col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.CENTER, PP_ALIGN.LEFT],
        bold_last_row=True,
    )

    add_subheading(s4, 0.70, 4.82, 5.75, "In Scope (Verified Benchmark)")
    tb4l = add_textbox(s4, 0.70, 5.18, 5.75, 1.75)
    add_bullets(tb4l.text_frame, [
        ("Repositories:", "5 public laboratory collections & unified SOH table."),
        ("Models & Protocols:", "Mean, Ridge, ElasticNet, XGBoost across P1–P4 (5 seeds)."),
        ("Reproducibility:", "6,120 executed runs + 320 logged infeasible settings."),
    ], font_size=12.5, space_after=6)

    add_subheading(s4, 6.88, 4.82, 5.75, "Out of Scope / Limitations")
    tb4r = add_textbox(s4, 6.88, 5.18, 5.75, 1.75)
    add_bullets(tb4r.text_frame, [
        ("No Fleet Telemetry:", "Laboratory cyclers only; no production BMS claims."),
        ("Sequence Code Excluded:", "LSTM / 1D-CNN / Transformer not in empirical scores."),
        ("No Rich V/I Claims:", "Cannot attribute transfer errors to any single physical factor."),
    ], font_size=12.5, space_after=6)

    # =========================================================================
    # SLIDE 5: 02 PROBLEM STATEMENT, MOTIVATION & OBJECTIVES
    # =========================================================================
    s5 = new_content_slide()
    add_template_header(
        s5, "Problem Statement",
        "Cross-Domain Formulation, Motivation & Six Core Objectives", page_num=5
    )
    add_subheading(s5, 0.70, 1.38, 5.75, "Formal Problem Definition & Motivation")
    tb5l = add_textbox(s5, 0.70, 1.76, 5.75, 5.05)
    add_bullets(tb5l.text_frame, [
        ("Problem Statement:", "When trained on source laboratory dataset S, how accurately do simple ML estimators predict capacity-derived SOHt = 100 × Qt / Qrated on unseen battery cells of target dataset T using only shared cycle index Xt = [ct]?"),
        ("Evaluation Metric:", "Observation-weighted Test MAE (SOH percentage points, pp) strictly across battery-disjoint partitions."),
        ("Prevent Leakage:", "Row-level random splits leak correlated cell trajectories; feeding present-cycle Qt leaks the target label."),
        ("Baseline Reference:", "Must test whether complex models actually outperform a naive Training-Mean predictor."),
    ], font_size=13.5, space_after=10)

    add_subheading(s5, 6.85, 1.38, 5.75, "Six Project Objectives (§2.3)")
    tb5r = add_textbox(s5, 6.85, 1.76, 5.75, 5.05)
    add_bullets(tb5r.text_frame, [
        ("Canonical Table:", "Harmonize capacity events from all 5 public repositories preserving source provenance."),
        ("Clean Label Derivation:", "Compute SOH using source Qrated and forbid capacity/SOH predictor leakage."),
        ("Whole-Cell Splitting:", "Enforce strict battery-disjoint train, validation, support & test sets."),
        ("Comprehensive Benchmark:", "Compare Mean, Ridge, ElasticNet & XGBoost across P1–P4 (5 seeds)."),
        ("Evaluate Adaptation:", "Quantify labelled k-shot support cells vs. label-free CORAL / normalization."),
        ("Honest Disclosure:", "Report data caveats, span mismatch & all 320 infeasible runs."),
    ], font_size=13.5, space_after=8)

    # =========================================================================
    # SLIDE 6: 03 LITERATURE SURVEY AND RESEARCH GAPS
    # =========================================================================
    s6 = new_content_slide()
    add_template_header(
        s6, "Literature Survey and Research Gaps",
        "Synthesis of Classical SOH, Transfer Learning & Leakage Literature", page_num=6
    )
    add_booktabs_table(
        s6, 0.70, 1.38, 11.93, 2.72,
        [2.25, 3.15, 2.75, 3.78],
        ["Research Strand", "Representative Citations", "Core Methodology", "Gap in Cross-Dataset Generalization"],
        [
            ["Physics & Filtering", "Plett [15] (2004); Berecibar [1] (2016); Hu [2] (2020)", "Coulomb counting, EKF & ECM parameterization", "Needs cell-specific parameters; weak chemistry portability."],
            ["Data-Driven & Transfer", "Severson [4]; Roman [5]; Ng [6]; Zhang [7]; Tan [8]; Wang [11]", "Early voltage features, RF/GP, CNN/LSTM & PINN transfer", "Evaluated within single labs or with unshared sensor signals."],
            ["Alignment & Rigor", "Sun CORAL [17]; Gulrajani [18]; Kaufman [13]; Kapoor [14]; Tan [12]", "Second-order covariance alignment; leakage audits", "Revealed split leakage & lack of trivial mean baselines."],
        ],
        font_size=11.5, header_size=12.0,
    )

    add_subheading(s6, 0.70, 4.35, 11.93, "Four Research Gaps Addressed in This Study")
    tb6 = add_textbox(s6, 0.70, 4.74, 11.93, 2.15)
    add_bullets(tb6.text_frame, [
        ("1. Within-Cell & Target Leakage:", "Solved via strict whole-cell disjoint splits and prohibiting present-cycle capacity/SOH inputs."),
        ("2. Missing Trivial Baselines:", "Solved by reporting Training-Mean / Target-Mean errors across every experimental protocol."),
        ("3. Inconsistent Feature Schemas:", "Solved via an honest cycle-index boundary instead of fabricating unshared voltage/temp signals."),
        ("4. Undocumented Pool Failures:", "Solved by explicitly logging all 320 infeasible small-pool experiments (NASA & Oxford)."),
    ], font_size=13.5, space_after=7)

    # =========================================================================
    # SLIDE 7: 04 DETAILED DESIGN — ARCHITECTURE & PROTOCOLS
    # =========================================================================
    s7 = new_content_slide()
    add_template_header(
        s7, "Detailed Design — Architecture & Protocols",
        "Modular Pipeline Architecture, Fixed Models & Evaluation Protocols (Tables 5.1 & 5.2)", page_num=7
    )
    add_subheading(s7, 0.70, 1.36, 5.85, "System Architecture (Fig. 5.1)")
    add_fit_picture(s7, ASSETS / "fig5_1_architecture.png", 0.65, 1.72, 5.95, 3.05)

    tb7 = add_textbox(s7, 0.70, 4.92, 5.85, 1.95)
    add_bullets(tb7.text_frame, [
        ("Fixed Hyperparameters:", "All models fixed in advance; zero test-set tuning."),
        ("Five Random Seeds:", "Seeds {11, 22, 33, 44, 55} control cell splits & XGBoost."),
        ("Label-Free CORAL / Norm:", "Affine rescaling of ct (λ = 1e-3) without target labels."),
    ], font_size=12.5, space_after=6)

    add_subheading(s7, 6.85, 1.36, 5.78, "Experimental Protocols (Table 5.1)")
    add_booktabs_table(
        s7, 6.85, 1.72, 5.78, 2.38,
        [1.20, 2.92, 1.66],
        ["Protocol", "Split & Training Regime", "Executed Runs"],
        [
            ["P1 Within", "25% test, 20% rem. val, rest train", "5-seed summaries*"],
            ["P2 Transfer", "Source train → all target cells (20 pairs)", "500 runs"],
            ["P3 LODO", "4 pooled train splits → held-out 5th lab", "100 runs"],
            ["P4 Few-Shot", "k∈{0,1,3,5,10} support + 30% disjoint test", "5,520 (+320 inf.)"],
            ["Total Reported", "4 Models × 5 Seeds (11..55)", "6,120 executed"],
        ],
        font_size=10.8, header_size=11.0, bold_last_row=True,
        col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT],
    )

    add_subheading(s7, 6.85, 4.24, 5.78, "Model Configurations (Table 5.2)")
    add_booktabs_table(
        s7, 6.85, 4.60, 5.78, 2.15,
        [1.30, 2.88, 1.60],
        ["Model", "Fixed Parameters", "Purpose"],
        [
            ["Mean", "Constant training-label average", "Reference baseline"],
            ["Ridge", "Standardized ct, L2 α = 1.0", "Linear slope model"],
            ["ElasticNet", "α = 0.01, L1 ratio = 0.5", "Regularized linear"],
            ["XGBoost", "150 trees, depth 4, lr 0.05, sub 0.8/0.9", "Nonlinear tree ens."],
        ],
        font_size=10.8, header_size=11.0,
    )

    # =========================================================================
    # SLIDE 8: 04 DETAILED DESIGN — CLASS & ACTIVITY DIAGRAMS
    # =========================================================================
    s8 = new_content_slide()
    add_template_header(
        s8, "Detailed Design — Class & Activity Flows",
        "Conceptual Software Structure & Quality-Gate Execution Flow (Figs. 5.2, 5.3 & 6.1)", page_num=8
    )
    add_subheading(s8, 0.70, 1.38, 5.75, "Conceptual Class Diagram (Fig. 5.2)")
    add_fit_picture(s8, ASSETS / "fig5_2_class_diagram.jpg", 0.65, 1.78, 5.85, 3.00)
    tb8l = add_textbox(s8, 0.70, 4.88, 5.75, 1.25)
    add_bullets(tb8l.text_frame, [
        ("Isolation:", "DatasetLoader encapsulates messy lab file formats."),
        ("Disjoint Guards:", "Harmonizer & SplitManager prevent cell overlap."),
    ], font_size=12.5, space_after=6)

    add_subheading(s8, 6.85, 1.38, 5.75, "Preparation & Quality Gate (Fig. 5.3)")
    add_fit_picture(s8, ASSETS / "fig5_3_activity_diagram.jpg", 6.80, 1.78, 5.85, 3.00)
    tb8r = add_textbox(s8, 6.85, 4.88, 5.75, 1.25)
    add_bullets(tb8r.text_frame, [
        ("Quality Gate:", "Invalid capacity/IDs force parser repair pre-benchmark."),
        ("Reproducibility:", "Saves logs, metrics CSVs, predictions & figures."),
    ], font_size=12.5, space_after=6)

    add_hline(s8, 0.70, 6.18, 11.93, color=RULE_LIGHT, thickness_pt=0.75)
    add_textbox(
        s8, 0.70, 6.26, 11.93, 0.55,
        "Research Use Case (Fig. 6.1):  Researcher selects dataset/protocol  →  Inspects/prepares unified table  →  Executes benchmark  →  Reviews outputs or optional read-only dashboard.",
        font_size=12.5, italic=True, color=BMS_BLUE, align=PP_ALIGN.CENTER
    )

    # =========================================================================
    # SLIDE 9: 05 IMPLEMENTATION
    # =========================================================================
    s9 = new_content_slide()
    add_template_header(
        s9, "Implementation",
        "Tools, Technologies & Source-Data Engineering Remediation (§6.1–6.3)", page_num=9
    )
    add_subheading(s9, 0.70, 1.36, 4.35, "Implementation Stack (Table 6.1)")
    tb9 = add_textbox(s9, 0.70, 1.76, 4.35, 5.05)
    add_bullets(tb9.text_frame, [
        ("Core Python CLI:", "Modular commands for inspect, prepare, matrix, lodo & adaptation."),
        ("Source Readers:", "SciPy (MATLAB v5: NASA/Oxford), h5py (HDF5 v7.3: Oxford/MATR), Pickle (XJTU), Excel & TSV (CALCE)."),
        ("Classical ML:", "scikit-learn & XGBoost on CPU; PyTorch DL modules optional/excluded."),
        ("Archive Policy:", "Raw archives (~11.7 GB) stay external; ODbL 1.0 (Oxford) & source attributions documented."),
    ], font_size=12.5, space_after=9)

    add_subheading(s9, 5.25, 1.36, 7.40, "Real-World Data Challenges & Fixes Applied (§6.3)")
    add_booktabs_table(
        s9, 5.25, 1.76, 7.40, 4.95,
        [1.50, 2.75, 3.15],
        ["Challenge", "Root Anomaly Observed", "Implemented Mitigation"],
        [
            ["CALCE CADEX", '"Capacity" is progress %, not Ah', "Integrated |I|·dt over elapsed time (CS2_8 & CS2_21)."],
            ["CALCE Arbin (13)", "Bundled discharge & resetting counters", "Segmented discharge steps & re-indexed cycles monotonically."],
            ["Oxford Sparse", "Checked once per ~100 drive cycles", "Retained exact 519 checkpoints; zero label interpolation."],
            ["MATR Batch IDs", "3 files, 4 spikes >1.43 Ah, 140 vs 124", "Canonicalized IDs, removed 4 spikes, reported 140 count."],
            ["49× Span Shift", "Max cycle ranges 168 to 8,200", "Retained raw cycles & diagnosed linear extrapolation."],
            ["Small Cell Pools", "NASA (4) & Oxford (8) limit high k", "Logged all 320 infeasible runs rather than resampling."],
        ],
        font_size=11.0, header_size=11.5,
    )

    # =========================================================================
    # SLIDE 10: 06 TESTING
    # =========================================================================
    s10 = new_content_slide()
    add_template_header(
        s10, "Testing",
        "Four-Layer Verification Evidence & Data Quality Audit (§7.1)", page_num=10
    )
    add_subheading(s10, 0.70, 1.36, 6.70, "Test Evidence Summary (Table 6.2)")
    add_booktabs_table(
        s10, 0.70, 1.72, 6.70, 3.65,
        [1.45, 2.55, 2.70],
        ["Evidence Layer", "Documented Evidence", "Interpretation"],
        [
            ["Unit / Fixture", "30 passed, 1 skipped (pytest README)", "Verifies software logic & parsers (not model accuracy)."],
            ["Real Parser Checks", "13 CALCE Arbin; CS2_8 & CS2_21; XJTU schema", "Source-format validation on real laboratory files."],
            ["Pipeline Integration", "Unified 5-lab table (190 cells, 132,389 rows)", "Zero cell leakage across train/val/support/test."],
            ["System Benchmark", "6,120 executed runs + 320 infeasible logged", "5-seed workload completed with zero runtime errors."],
        ],
        font_size=11.0, header_size=11.5,
    )

    tb10note = add_textbox(s10, 0.70, 5.55, 6.70, 1.25)
    add_bullets(tb10note.text_frame, [
        ("Strict Distinction:", "Software unit tests verify code behavior on fixtures; benchmark accuracy comes exclusively from the 6,120 real-data experiments. One baseline test is recorded as skipped."),
    ], font_size=12.5, space_after=4)

    add_subheading(s10, 7.65, 1.36, 5.00, "Dataset Quality Audit (data_quality.csv)")
    add_booktabs_table(
        s10, 7.65, 1.72, 5.00, 3.20,
        [0.90, 0.65, 1.00, 1.15, 1.30],
        ["Dataset", "Cells", "Obs.", "Max Cyc.", "Status"],
        [
            ["CALCE", "15", "9,608", "1,308", "PASS (15 EOL)"],
            ["MATR", "140", "114,688", "2,237", "PASS (43 EOL)"],
            ["NASA", "4", "636", "168", "PASS (4 EOL)"],
            ["Oxford", "8", "519", "8,200", "PASS (8 EOL)"],
            ["XJTU", "23", "6,938", "419", "PASS (4 EOL)"],
            ["Total", "190", "132,389", "8,200", "ALL PASSED"],
        ],
        font_size=10.8, header_size=11.0,
        col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.CENTER],
        bold_last_row=True,
    )
    tb10r = add_textbox(s10, 7.65, 5.12, 5.00, 1.65)
    add_bullets(tb10r.text_frame, [
        ("Whole-Cell Keys:", "Partitioned strictly on dataset:battery_id."),
        ("Zero Target Leakage:", "Qt & SOHt excluded from predictors."),
        ("Raw Tails Preserved:", "No label smoothing or EOL truncation."),
    ], font_size=12.0, space_after=5)

    # =========================================================================
    # SLIDE 11: 07 RESULTS — WITHIN VS. ZERO-SHOT PAIRWISE TRANSFER
    # =========================================================================
    s11 = new_content_slide()
    add_template_header(
        s11, "Results — Within-Dataset vs. Zero-Shot Transfer",
        "Within-Domain Skill Does Not Predict Zero-Shot Transferability (5×5 Matrix, 500 Runs)", page_num=11
    )
    add_fit_picture(s11, ASSETS / "fig3_transfer_heatmap.png", 0.65, 1.38, 7.50, 3.15)

    tb11l = add_textbox(s11, 0.70, 4.68, 7.45, 2.15)
    add_bullets(tb11l.text_frame, [
        ("Training Mean Leads Overall:", "Mean beats XGBoost in 12 of 20 pairs (11.22 vs. 13.03 pp); XGBoost wins 8 pairs, Ridge 6."),
        ("Non-CALCE Check (12 pairs):", "Ranking holds without noisy CALCE — Mean 9.57 pp vs. XGBoost 12.01 vs. Ridge 79.04."),
        ("Oxford → MATR (2.73–2.87 pp):", "Incidental curve alignment over cycles 1–2,000; not causal chemistry transfer."),
    ], font_size=12.0, space_after=6)

    add_subheading(s11, 8.32, 1.34, 4.35, "Within-Dataset MAE (pp, 5 Seeds)")
    add_booktabs_table(
        s11, 8.32, 1.70, 4.35, 2.78,
        [0.95, 1.13, 1.13, 1.14],
        ["Dataset", "Mean", "Ridge", "XGBoost"],
        [
            ["NASA", "9.09±2.09", "4.41±1.60", "4.29±1.39"],
            ["CALCE", "16.71±1.67", "11.25±2.06", "9.40±2.20"],
            ["XJTU", "3.93±0.07", "2.39±0.20", "1.82±0.25"],
            ["Oxford", "6.04±0.29", "1.72±0.50", "1.25±0.35"],
            ["MATR", "3.58±0.17", "2.77±0.20", "2.71±0.22"],
            ["All (25)", "7.87±5.06", "4.51±3.72", "3.89±3.19"],
        ],
        font_size=10.2, header_size=10.5,
        col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER],
        bold_last_row=True,
        bold_cells={(r, 3) for r in range(5)},
    )

    add_subheading(s11, 8.32, 4.62, 4.35, "Zero-Shot Average (20 Pairs)")
    add_booktabs_table(
        s11, 8.32, 4.98, 4.35, 1.78,
        [1.35, 1.70, 1.30],
        ["Model", "MAE ± SD (pp)", "Wins"],
        [
            ["Mean", "11.22 ± 4.24", "12 / 20 pairs"],
            ["XGBoost", "13.03 ± 8.45", "8 / 20 pairs"],
            ["ElasticNet", "61.31 ± 138.63", "≤ 6 / 20 pairs"],
            ["Ridge", "61.49 ± 138.98", "6 / 20 pairs"],
        ],
        font_size=10.5, header_size=10.5,
        bold_cells={(0, 1)},
        warn_cells={(2, 1), (3, 1)},
    )

    # =========================================================================
    # SLIDE 12: 07 RESULTS — LINEAR EXTRAPOLATION FAILURE
    # =========================================================================
    s12 = new_content_slide()
    add_template_header(
        s12, "Results — Linear Extrapolation Failure Diagnosis",
        "Transferring Short-Span Calendar Slopes to 8,200-Cycle Targets Produces Catastrophic Divergence (Fig. 4)", page_num=12
    )
    add_fit_picture(s12, ASSETS / "fig4_extrapolation.png", 0.90, 1.36, 11.53, 4.18)

    tb12 = add_textbox(s12, 0.70, 5.64, 11.93, 1.32)
    add_bullets(tb12.text_frame, [
        ("NASA → Oxford Divergence (637.45 ± 75.52 pp Ridge):", "Fitted on NASA ≤168 cycles (slope −0.196 %/cyc), Ridge predicts −1,511% SOH at cycle 8,200; XGBoost clamps to terminal leaf (~65.9% → 17.10 pp MAE)."),
        ("Cycle-Range Correlation (3 Pairs >100 pp):", "All 3 Ridge blowups target Oxford (NASA 637.5, CALCE 133.8, XJTU 110.5 pp). Max-cycle ratio correlates with Ridge error (Spearman ρ = 0.55, p = 0.012)."),
    ], font_size=12.5, space_after=6)

    # =========================================================================
    # SLIDE 13: 07 RESULTS — LEAVE-ONE-DATASET-OUT (LODO)
    # =========================================================================
    s13 = new_content_slide()
    add_template_header(
        s13, "Results — Leave-One-Dataset-Out (LODO)",
        "Four-Source Pooling Prevents Extrapolation Collapse Without Beating Mean Baseline (Table IV)", page_num=13
    )
    add_subheading(s13, 0.70, 1.36, 6.40, "LODO Held-Out Test MAE (pp, Mean ± SD, 100 Runs)")
    add_booktabs_table(
        s13, 0.70, 1.72, 6.40, 3.40,
        [1.10, 0.90, 1.46, 1.47, 1.47],
        ["Held-Out", "Train Cells", "Mean", "Ridge", "XGBoost"],
        [
            ["NASA", "112", "13.95 ± 0.04", "16.16 ± 0.36", "17.34 ± 0.20"],
            ["CALCE", "105", "16.29 ± 0.05", "15.69 ± 0.11", "15.38 ± 0.15"],
            ["XJTU", "100", "4.17 ± 0.07", "3.29 ± 0.06", "3.60 ± 0.08"],
            ["Oxford", "109", "7.69 ± 0.03", "19.08 ± 1.34", "5.88 ± 0.67"],
            ["MATR", "30", "9.13 ± 1.12", "9.40 ± 1.25", "11.92 ± 0.75"],
            ["All (25)", "30–112", "10.25 ± 4.47", "12.72 ± 5.84", "10.82 ± 5.44"],
        ],
        font_size=11.0, header_size=11.0,
        col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER],
        bold_last_row=True,
        bold_cells={(0, 2), (1, 4), (2, 3), (3, 4), (4, 2)},
    )

    tb13l = add_textbox(s13, 0.70, 5.26, 6.40, 1.60)
    add_bullets(tb13l.text_frame, [
        ("Worst-Case Error Bounded:", "Largest LODO MAE drops to 19.08 pp (vs. 637.45 pp zero-shot Ridge). ElasticNet = 12.69 ± 5.80 pp overall."),
        ("MATR Dominance Caveat:", "MATR contributes 86.6% of all rows (84/105 train cells for CALCE), so LODO reflects training dominated by MATR."),
    ], font_size=12.0, space_after=6)

    add_subheading(s13, 7.35, 1.36, 5.25, "Cross-Protocol MAE Comparison (Fig. 6.2)")
    add_fit_picture(s13, ASSETS / "fig6_2_lodo_summary.png", 7.30, 1.76, 5.33, 3.05)

    tb13r = add_textbox(s13, 7.35, 4.98, 5.25, 1.85)
    add_bullets(tb13r.text_frame, [
        ("Target Wins:", "Models beat Mean on Oxford (XGBoost 5.88 vs. 7.69), XJTU (Ridge 3.29) & CALCE (15.38)."),
        ("Overall Aggregate:", "Training Mean (10.25 pp) remains ahead of XGBoost (10.82 pp) & wins on NASA/MATR."),
    ], font_size=12.0, space_after=6)

    # =========================================================================
    # SLIDE 14: 07 RESULTS — FEW-SHOT ADAPTATION
    # =========================================================================
    s14 = new_content_slide()
    add_template_header(
        s14, "Results — Few-Shot Adaptation & CORAL Alignment",
        "Target Support Labels Improve XGBoost From k = 3 Onward; Label-Free Alignment Fails (Fig. 5 & Table V)", page_num=14
    )
    add_fit_picture(s14, ASSETS / "fig5_fewshot.png", 0.65, 1.38, 7.30, 3.65)

    tb14l = add_textbox(s14, 0.70, 5.14, 7.25, 1.75)
    add_bullets(tb14l.text_frame, [
        ("Infeasible Settings Logged (320):", "NASA feasible only at k=1; Oxford up to k=5 because target pools are small."),
        ("Methodological Correction (§7.2):", "Earlier draft claim (k=1 sufficient) is superseded: at k=1, no model beats Target Mean (8.53 pp). Reliable gains emerge from k = 3 on fixed targets (CALCE, MATR, XJTU)."),
    ], font_size=11.8, space_after=5)

    add_subheading(s14, 8.12, 1.36, 4.56, "Table V: Fixed Common Targets (pp)")
    add_booktabs_table(
        s14, 8.12, 1.72, 4.56, 3.30,
        [1.76, 0.56, 0.56, 0.56, 0.56, 0.56],
        ["Method", "k=0", "k=1", "k=3", "k=5", "k=10"],
        [
            ["Target Mean", "11.84", "8.53", "8.11", "8.21", "8.03"],
            ["Ridge scratch", "21.89", "21.46", "6.55", "6.02", "5.69"],
            ["Ridge pooled", "21.89", "12.29", "8.02", "7.48", "6.98"],
            ["XGB scratch", "12.11", "10.78", "6.22", "5.12", "4.98"],
            ["XGB pooled", "12.11", "10.00", "7.52", "6.54", "5.96"],
            ["XGB CORAL", "12.11", "12.88", "11.02", "10.90", "10.82"],
            ["XGB M-V Norm", "12.11", "12.88", "11.02", "10.90", "10.82"],
        ],
        font_size=10.0, header_size=10.2,
        col_aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 5,
        bold_cells={(0, 2), (3, 3), (3, 4), (3, 5)},
    )

    tb14r = add_textbox(s14, 8.12, 5.14, 4.56, 1.75)
    add_bullets(tb14r.text_frame, [
        ("Why CORAL Fails Here:", "Stays at 10.8–12.9 pp (never beating k=0 = 12.11). With 1 feature (ct), affine rescaling has no target SOH labels (mean diff 0.01 pp)."),
    ], font_size=11.8, space_after=4)

    # =========================================================================
    # SLIDE 15: 07 RESULTS — OUTPUT SNAPSHOTS
    # =========================================================================
    s15 = new_content_slide()
    add_template_header(
        s15, "Results and Snapshots of the Output",
        "Verified Repository Structure & Saved Benchmark Plot Artifacts (Appendix A)", page_num=15
    )
    add_subheading(s15, 0.70, 1.36, 5.80, "A. Verified Repository & Run Manifest")
    tb15a = add_textbox(s15, 0.70, 1.74, 5.80, 2.10)
    add_bullets(tb15a.text_frame, [
        ("Core Package:", "battery_soh/ {data, models, experiments, evaluation}."),
        ("Unified Data:", "data/processed/unified.csv (132,389 rows) & data_quality.csv."),
        ("Study Outputs:", "results/full_study/ — matrix (500), lodo (100), adapt (5,520 + 320 inf.)."),
        ("Verification:", "30 passed, 1 skipped (pytest); Seeds {11, 22, 33, 44, 55}."),
    ], font_size=11.8, space_after=5)

    add_fit_picture(s15, RESULTS / "matrix" / "figures" / "ranking.png", 6.80, 1.38, 5.80, 2.22)
    add_textbox(s15, 6.80, 3.62, 5.80, 0.28, "B. Model MAE Ranking Over Matrix Runs (results/full_study/matrix/)", font_size=11.0, italic=True, color=MUTED, align=PP_ALIGN.CENTER)

    add_fit_picture(s15, RESULTS / "lodo" / "figures" / "lodo_target_comparison.png", 0.70, 4.02, 5.80, 2.48)
    add_textbox(s15, 0.70, 6.54, 5.80, 0.28, "C. Saved LODO Per-Target MAE Comparison Across 5 Labs", font_size=11.0, italic=True, color=MUTED, align=PP_ALIGN.CENTER)

    add_fit_picture(s15, RESULTS / "matrix" / "figures" / "transfer_ridge.png", 6.80, 4.02, 5.80, 2.48)
    add_textbox(s15, 6.80, 6.54, 5.80, 0.28, "D. Saved Ridge 5×5 Matrix Exposing Oxford Blowups (637.5 pp)", font_size=11.0, italic=True, color=MUTED, align=PP_ALIGN.CENTER)

    # =========================================================================
    # SLIDE 16: 08 CONCLUSION
    # =========================================================================
    s16 = new_content_slide()
    add_template_header(
        s16, "Conclusion",
        "Key Empirical Findings & Recommended Reporting Standard", page_num=16
    )
    add_subheading(s16, 0.70, 1.38, 11.93, "Primary Experimental Conclusions")
    tb16a = add_textbox(s16, 0.70, 1.76, 11.93, 2.45)
    add_bullets(tb16a.text_frame, [
        ("Within-Domain Mirage:", "XGBoost achieves 3.89 ± 3.19 pp MAE inside datasets, yet zero-shot transfer falls to 13.03 ± 8.45 pp — failing to beat the simple Training Mean (11.22 ± 4.24 pp; Mean wins 12/20 pairs)."),
        ("Extrapolation Breakdown:", "Unconstrained linear models diverge catastrophically under cycle-span mismatch (up to 637.45 ± 75.52 pp for NASA → Oxford; Spearman ρ = 0.55, p = 0.012)."),
        ("LODO Pooling:", "Pooling four sources bounds maximum error to 19.08 pp, yet remains MATR-dominated (86.6% of rows) and trails the Mean overall (10.82 vs. 10.25 pp)."),
        ("Labels Beat 1D Alignment:", "Target support labels improve XGBoost clearly from k = 3 onward (6.22 → 4.98 pp), whereas label-free CORAL/norm fail (10.8–12.9 pp)."),
    ], font_size=13.5, space_after=8)

    add_hline(s16, 0.70, 4.35, 11.93, color=BMS_BLUE, thickness_pt=1.0)
    add_subheading(s16, 0.70, 4.48, 11.93, "Recommended Reporting Standard for Cross-Dataset SOH Studies")
    tb16b = add_textbox(s16, 0.70, 4.86, 11.93, 2.00)
    add_bullets(tb16b.text_frame, [
        ("Enforce Cell-Disjoint Splits:", "Always separate whole cells across train/test and exclude current-cycle capacity/SOH inputs."),
        ("Report a Mean Reference & Span Ratio:", "Compare every transferred model against Training Mean and disclose target/source cycle ratios."),
        ("Preserve Infeasible Settings & Weighted Caveats:", "Document small-pool failures (320 runs) and separate observation-weighted vs. per-cell metrics."),
    ], font_size=13.5, space_after=8)

    # =========================================================================
    # SLIDE 17: 09 FUTURE ENHANCEMENTS
    # =========================================================================
    s17 = new_content_slide()
    add_template_header(
        s17, "Future Enhancements",
        "Seven Prioritized Extensions Toward Fleet-Ready BMS Transfer (Conclusion, p. 23)", page_num=17
    )
    tb17 = add_textbox(s17, 0.85, 1.48, 11.63, 5.40)
    add_bullets(tb17.text_frame, [
        ("1. Synchronized Physical Feature Extraction:", "Re-extract comparable voltage, current, temperature & partial-charge (dQ/dV) features uniformly from all five raw repositories."),
        ("2. MATR Count Sensitivity Analysis:", "Validate parser labels and investigate the 140-parsed versus 124 original-study MATR cell discrepancy."),
        ("3. CALCE Low-SOH & Metadata Audit:", "Investigate CALCE sub-50% capacity observations (incomplete discharge segments) and rated-capacity assumptions."),
        ("4. Per-Cell Macro Metrics & Intervals:", "Complement observation-weighted MAE with per-cell macro averages and formal uncertainty intervals."),
        ("5. Independent Vehicle & Fleet Validation:", "Evaluate models on operational real-world EV fleet logs beyond controlled laboratory test benches."),
        ("6. Calibrated & Physically Constrained Models:", "Bound predictions inside physical SOH intervals and explore physics-informed constraints without leakage."),
        ("7. Sequence Models & Multi-Feature CORAL:", "Assess implemented PyTorch architectures (LSTM, 1D-CNN, Transformer) & feature-level alignment once steps 1–6 are complete."),
    ], font_size=15.0, space_after=12, line_spacing=1.20)

    # =========================================================================
    # SLIDE 18: 10 PAPER PUBLISHED / SUBMISSION STATUS
    # =========================================================================
    s18 = new_content_slide()
    add_template_header(
        s18, "Research Paper Submission Status",
        "Manuscript Submitted to Journal of Energy Storage (Elsevier) — Appendix C", page_num=18
    )
    add_subheading(s18, 0.70, 1.36, 5.55, "Submission Metadata (Verified)")
    add_booktabs_table(
        s18, 0.70, 1.74, 5.55, 4.75,
        [1.65, 3.90],
        ["Attribute", "Submission Record"],
        [
            ["Paper Title", "Cross-Dataset Generalization and Few-Shot Adaptation of Machine-Learning Models for Electric Vehicle Battery State-of-Health Estimation"],
            ["Journal", "Journal of Energy Storage (Elsevier)"],
            ["Manuscript ID", "EST-D-26-20198"],
            ["Date Submitted", "10 November 2026 (2:18 PM)"],
            ["Current Status", "Submissions Being Processed (confirms submission, NOT acceptance/publication)"],
            ["Authors", "O. M. Makadia, D. Jain, Goutham T. G., D. Pranav Sai, Dr. M. Dakshayini"],
            ["Repository", "github.com/om-makadia-118/ev-battery-soh"],
        ],
        font_size=11.0, header_size=11.5,
        bold_cells={(2, 1)},
    )

    add_subheading(s18, 6.50, 1.36, 6.13, "Editorial Manager Confirmation Proof (Fig. C.1)")
    add_fit_picture(s18, ASSETS / "email_proof_EST_D_26_20198.png", 6.50, 1.78, 6.13, 4.45)
    add_textbox(
        s18, 6.50, 6.32, 6.13, 0.45,
        "Fig. C.1: Automated submission confirmation from Elsevier Editorial Manager (EST-D-26-20198).",
        font_size=11.0, italic=True, color=MUTED, align=PP_ALIGN.CENTER
    )

    # =========================================================================
    # SLIDE 19: 11 REFERENCES
    # =========================================================================
    s19 = new_content_slide()
    add_template_header(
        s19, "References",
        "Primary Literature & Public Battery Datasets Cited ([1]–[21])", page_num=19
    )
    refs_left = [
        "[1] M. Berecibar et al., “Critical review of state of health estimation methods of Li-ion batteries for real applications,” Renew. Sustain. Energy Rev., vol. 56, pp. 572–587, 2016.",
        "[2] X. Hu, L. Xu, X. Lin, and M. Pecht, “Battery lifetime prognostics,” Joule, vol. 4, no. 2, pp. 310–346, 2020.",
        "[3] Y. Li et al., “Data-driven health estimation and lifetime prediction of lithium-ion batteries: A review,” Renew. Sustain. Energy Rev., vol. 113, Art. 109254, 2019.",
        "[4] K. A. Severson et al., “Data-driven prediction of battery cycle life before capacity degradation,” Nature Energy, vol. 4, pp. 383–391, 2019. (MATR Dataset)",
        "[5] D. Roman et al., “Machine learning pipeline for battery state-of-health estimation,” Nat. Mach. Intell., vol. 3, pp. 447–456, 2021.",
        "[6] M.-F. Ng et al., “Predicting state of charge and health of batteries using data-driven ML,” Nat. Mach. Intell., vol. 2, pp. 161–170, 2020.",
        "[7] Y. Zhang et al., “Identifying degradation patterns of Li-ion batteries from EIS,” Nature Communications, vol. 11, Art. 1706, 2020.",
        "[8] Y. Tan & G. Zhao, “Transfer learning with LSTM network for SOH prediction,” IEEE Trans. Ind. Electron., vol. 67, pp. 8723–8731, 2020.",
        "[9] S. Shen et al., “Deep CNNs with ensemble & transfer learning for capacity estimation,” Applied Energy, vol. 260, Art. 114296, 2020.",
        "[10] Z. Ren et al., “SOH estimation based on deep neural network & transfer learning,” Batteries, vol. 9, Art. 585, 2023.",
    ]
    refs_right = [
        "[11] F. Wang et al., “Physics-informed neural network for Li-ion battery degradation modeling,” Nature Communications, vol. 15, Art. 4332, 2024. (XJTU)",
        "[12] R. Tan et al., “BatteryLife: Comprehensive dataset & benchmark for battery life prediction,” Proc. 31st ACM SIGKDD, pp. 5789–5800, 2025.",
        "[13] S. Kaufman et al., “Leakage in data mining: Formulation, detection, and avoidance,” ACM TKDD, vol. 6, Art. 15, 2012.",
        "[14] S. Kapoor & A. Narayanan, “Leakage and the reproducibility crisis in ML-based science,” Patterns, vol. 4, Art. 100804, 2023.",
        "[15] G. L. Plett, “Extended Kalman filtering for BMS of LiPB-based HEV battery packs,” J. Power Sources, vol. 134, pp. 252–261, 2004.",
        "[16] S. J. Pan & Q. Yang, “A survey on transfer learning,” IEEE Trans. Knowl. Data Eng., vol. 22, pp. 1345–1359, 2010.",
        "[17] B. Sun, J. Feng, & K. Saenko, “Return of frustratingly easy domain adaptation (CORAL),” Proc. AAAI, pp. 2058–2065, 2016.",
        "[18] I. Gulrajani & D. Lopez-Paz, “In search of lost domain generalization,” Proc. ICLR, 2021.",
        "[19] B. Saha & K. Goebel, “Battery Data Set,” NASA Ames Prognostics Data Repository, 2007.",
        "[20–21] W. He et al., CALCE CS2 Battery Data (UMD, 2011); C. R. Birkl, Oxford Battery Degradation Dataset 1 (Univ. of Oxford, 2017).",
    ]
    for rx, rlist, hdr in [(0.70, refs_left, "Battery SOH & Transfer Literature"), (6.85, refs_right, "Benchmarks, Leakage & Datasets")]:
        add_subheading(s19, rx, 1.34, 5.78, hdr, size=15.0)
        tb = add_textbox(s19, rx, 1.72, 5.78, 5.18)
        tf = tb.text_frame
        for idx, line in enumerate(rlist):
            p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
            p.space_after = Pt(5)
            p.line_spacing = 1.10
            run = p.add_run()
            run.text = line
            run.font.name = FONT_TNR
            run.font.size = Pt(10.5)
            run.font.color.rgb = BLACK

    # =========================================================================
    # REORDER sldIdLst: [Title, Agenda, ...Content Slides 3..19..., Starter Q&A, Starter Thank You!]
    # =========================================================================
    # Remove qa_sldId and ty_sldId from positions 2, 3 and append cleanly at the end!
    sldIdLst.remove(qa_sldId)
    sldIdLst.remove(ty_sldId)
    sldIdLst.append(qa_sldId)
    sldIdLst.append(ty_sldId)

    OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PPTX))
    return OUT_PPTX


if __name__ == "__main__":
    out = build_deck()
    prs_check = Presentation(str(out))
    print(f"Built {out} ({out.stat().st_size:,} bytes, {len(prs_check.slides)} slides)")
