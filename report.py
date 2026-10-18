"""Builds the lab-style PDF report for the Lungwise lung cancer risk app.

build_report() returns the PDF as bytes, so the Streamlit app can hand it to st.download_button.
Nothing is written to disk and nothing is stored.
"""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, Image, KeepTogether,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

DEEP = colors.HexColor("#123B40")
TEAL = colors.HexColor("#1F8A84")
ROSE = colors.HexColor("#D1495B")
AMBER = colors.HexColor("#B7791F")
MUTED = colors.HexColor("#5B7176")
LINE = colors.HexColor("#DCE6E7")
MIST = colors.HexColor("#EEF4F4")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

BASE = dict(fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=DEEP)
S = {
    "body": ParagraphStyle("body", **BASE),
    "small": ParagraphStyle("small", **{**BASE, "fontSize": 8, "leading": 11, "textColor": MUTED}),
    "label": ParagraphStyle("label", **{**BASE, "fontSize": 8, "leading": 10, "textColor": MUTED, "fontName": "Helvetica-Bold"}),
    "cellb": ParagraphStyle("cellb", **{**BASE, "fontName": "Helvetica-Bold"}),
    "h2": ParagraphStyle("h2", **{**BASE, "fontName": "Helvetica-Bold", "fontSize": 11, "leading": 14, "spaceBefore": 14, "spaceAfter": 6}),
    "brand": ParagraphStyle("brand", **{**BASE, "fontName": "Helvetica-Bold", "fontSize": 22, "leading": 24}),
    "title": ParagraphStyle("title", **{**BASE, "fontName": "Helvetica-Bold", "fontSize": 11, "leading": 14, "alignment": TA_RIGHT}),
    "subtitle": ParagraphStyle("subtitle", **{**BASE, "fontSize": 8.5, "leading": 11, "textColor": MUTED, "alignment": TA_RIGHT}),
    "th": ParagraphStyle("th", **{**BASE, "fontName": "Helvetica-Bold", "fontSize": 8.5, "textColor": colors.white}),
}

TIER = {
    "Higher than average": (ROSE, "#D1495B"),
    "Moderate risk": (AMBER, "#B7791F"),
    "Lower than average": (TEAL, "#1F8A84"),
}
TIER_TEXT = {
    "Higher than average": "Your answers closely match people with a higher chance of lung cancer in the training data.",
    "Moderate risk": "Several of your answers match people with lung cancer in the training data.",
    "Lower than average": "Your answers match people with a lower chance of lung cancer in the training data.",
}


class ScoreBar(Flowable):
    """Horizontal 0-100% bar with tick marks at the two tier cut-offs (50% and 75%)."""

    def __init__(self, prob, color, width):
        super().__init__()
        self.prob, self.color, self.width, self.height = prob, color, width, 30

    def draw(self):
        c, w = self.canv, self.width
        y = 14
        c.setFillColor(LINE)
        c.roundRect(0, y, w, 8, 4, stroke=0, fill=1)
        c.setFillColor(self.color)
        c.roundRect(0, y, max(w * self.prob, 8), 8, 4, stroke=0, fill=1)
        c.setStrokeColor(DEEP)
        c.setLineWidth(0.8)
        c.setFont("Helvetica", 7)
        c.setFillColor(MUTED)
        for pct in (0, 50, 75, 100):
            x = w * pct / 100
            if pct in (50, 75):
                c.line(x, y - 2, x, y + 10)
            c.drawCentredString(min(max(x, 6), w - 8), y - 10, f"{pct}%")


class NumberedCanvas(rl_canvas.Canvas):
    """Adds the footer with patient, report ID and 'Page x of y' on every page."""

    footer_text = ""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self._footer(total)
            super().showPage()
        super().save()

    def _footer(self, total):
        self.setStrokeColor(LINE)
        self.setLineWidth(0.6)
        self.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(MUTED)
        self.drawString(MARGIN, 10 * mm, self.footer_text)
        self.drawRightString(PAGE_W - MARGIN, 10 * mm, f"Page {self._pageNumber} of {total}")


def _p(text, style="body"):
    return Paragraph(text, S[style])


def _esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _chart(contrib):
    """Horizontal bar chart of each answer's effect on the score (percentage points)."""
    items = sorted(((k, v * 100) for k, v in contrib.items() if abs(v) > 1e-6), key=lambda kv: kv[1])
    if not items:
        return None
    h = max(1.8, 0.38 * len(items) + 0.9)
    fig, ax = plt.subplots(figsize=(6.6, h))
    ax.barh([k for k, _ in items], [v for _, v in items],
            color=["#D1495B" if v > 0 else "#1F8A84" for _, v in items], height=0.6)
    ax.axvline(0, color="#9AA7AA", linewidth=0.8)
    ax.set_xlabel("Effect on the score, in percentage points (red raises, teal lowers)", color="#5B7176", fontsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#DCE6E7")
    ax.tick_params(colors="#123B40", length=0, labelsize=8.5)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=CONTENT_W, height=CONTENT_W * h / 6.6)


def _influence(value):
    pts = value * 100
    if abs(pts) < 0.5:
        return _p('<font color="#5B7176">Little effect</font>')
    if pts > 0:
        return _p(f'<font color="#D1495B"><b>Raised score</b></font> (+{pts:.1f} pts)')
    return _p(f'<font color="#1F8A84"><b>Lowered score</b></font> ({pts:.1f} pts)')


def build_report(name, age, gender, parameters, contrib, prob, label, advice,
                 rule_applied, report_id, generated_at):
    """Return the report as PDF bytes.

    parameters: list of (parameter name, value as shown to the user); names should match the keys of contrib.
    contrib:    {parameter name: effect on the score in probability units (SHAP)}.
    """
    color, hexcol = TIER.get(label, (TEAL, "#1F8A84"))
    name = name.strip() or "Not provided"
    out = io.BytesIO()

    doc = BaseDocTemplate(out, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=16 * mm, bottomMargin=20 * mm,
                          title="Lung Cancer Risk Screening Report", author="Lungwise")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(MARGIN, 20 * mm, CONTENT_W, PAGE_H - 36 * mm,
                                                             leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)])])
    story = []

    # ---- header ----
    head = Table([[_p("Lungwise", "brand"),
                   [_p("LUNG CANCER RISK SCREENING REPORT", "title"),
                    _p("AI-assisted screening with explainable results (SHAP)", "subtitle")]]],
                 colWidths=[CONTENT_W * 0.4, CONTENT_W * 0.6])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "BOTTOM"), ("LINEBELOW", (0, 0), (-1, 0), 1.4, TEAL),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story += [head, Spacer(1, 10)]

    # ---- patient details ----
    info = [
        [_p("PATIENT NAME", "label"), _p(f"<b>{_esc(name)}</b>"), _p("REPORT ID", "label"), _p(report_id)],
        [_p("AGE / GENDER", "label"), _p(f"{age} years / {_esc(gender)}"), _p("DATE AND TIME", "label"), _p(generated_at)],
        [_p("INPUT TYPE", "label"), _p("Self-reported questionnaire"), _p("REPORT TYPE", "label"), _p("Screening (not diagnostic)")],
    ]
    t = Table(info, colWidths=[CONTENT_W * 0.17, CONTENT_W * 0.33, CONTENT_W * 0.17, CONTENT_W * 0.33])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, LINE), ("BACKGROUND", (0, 0), (0, -1), MIST),
                           ("BACKGROUND", (2, 0), (2, -1), MIST), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    story.append(t)

    # ---- result ----
    story.append(_p("Result summary", "h2"))
    ref = ("Lower than average: below 50%<br/>Moderate: 50% to 75%<br/>Higher than average: 75% and above")
    res = Table([[_p("Test", "th"), _p("Result", "th"), _p("Reference range", "th"), _p("Flag", "th")],
                 [_p("Lung cancer risk score (AI model)"),
                  _p(f'<font size="15" color="{hexcol}"><b>{prob:.0%}</b></font>'),
                  _p(ref, "small"),
                  _p(f'<font color="{hexcol}"><b>{label.upper()}</b></font>')]],
                colWidths=[CONTENT_W * 0.30, CONTENT_W * 0.16, CONTENT_W * 0.32, CONTENT_W * 0.22])
    res.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), DEEP), ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 7),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    story += [res, Spacer(1, 8), ScoreBar(prob, color, CONTENT_W), Spacer(1, 4)]
    story.append(_p("The score shows how closely your answers match patients in a synthetic practice dataset. "
                    "It is not a real-world probability of having lung cancer.", "small"))

    # ---- parameters ----
    story.append(_p("Parameters entered and their effect", "h2"))
    rows = [[_p("Parameter", "th"), _p("Reported value", "th"), _p("Effect on score", "th")]]
    for pname, pval in parameters:
        rows.append([_p(f"<b>{_esc(pname)}</b>"), _p(_esc(pval)), _influence(contrib.get(pname, 0.0))])
    pt = Table(rows, colWidths=[CONTENT_W * 0.25, CONTENT_W * 0.45, CONTENT_W * 0.30], repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), DEEP), ("GRID", (0, 0), (-1, -1), 0.5, LINE),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LEFTPADDING", (0, 0), (-1, -1), 8)]
    for i in range(2, len(rows), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), MIST))
    pt.setStyle(TableStyle(style))
    story += [pt, Spacer(1, 3),
              _p("pts = percentage points that this answer added to or removed from the score (SHAP values).", "small")]

    # ---- explanation chart ----
    chart = _chart(contrib)
    if chart is not None:
        story.append(KeepTogether([_p("Why the model gave this result", "h2"), chart]))

    # ---- interpretation ----
    interp = [_p("Interpretation and advice", "h2"), _p(f"<b>{label}.</b> {TIER_TEXT.get(label, '')}"), Spacer(1, 4), _p(_esc(advice))]
    if rule_applied:
        interp += [Spacer(1, 4), _p("This result was raised to Moderate by a medical rule for smoking history of 10 or more "
                                    "pack-years, because the practice dataset understates the effect of smoking.", "small")]
    interp += [Spacer(1, 6),
               _p("<b>See a doctor promptly, whatever this score says, if you have:</b> a cough that lasts more than three weeks "
                  "or changes, coughing up blood, ongoing chest pain, unexplained weight loss, or increasing breathlessness.")]
    story.append(KeepTogether(interp))

    # ---- about ----
    about = [_p("About this report", "h2"),
             _p("<b>Model:</b> Random Forest classifier (300 trees) with medical constraints, so no factor can lower the risk. "
                "<b>Data:</b> public Kaggle Lung Cancer Risk dataset, 50,000 records. "
                "<b>Validation:</b> 5-fold cross-validated ROC-AUC 0.76 and Brier score 0.187. "
                "<b>Explanations:</b> SHAP.", "body"),
             Spacer(1, 4),
             _p("<b>Limitations:</b> the dataset appears to be synthetic, the model has not been clinically validated, and "
                "important factors such as air pollution, occupation and imaging are not included.", "body")]
    story.append(KeepTogether(about))

    # ---- disclaimer ----
    box = Table([[_p("<b>Important notice.</b> This report comes from an educational AI screening tool. It is not a medical "
                     "diagnosis and does not replace an examination by a qualified doctor. Do not start, stop or change any "
                     "treatment based on it. Lungwise does not store your name or answers.", "body")]],
                colWidths=[CONTENT_W])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, color), ("BACKGROUND", (0, 0), (-1, -1), MIST),
                             ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                             ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10)]))
    story += [Spacer(1, 14), KeepTogether([box]), Spacer(1, 6), _p("*** End of report ***", "small")]

    class _Canvas(NumberedCanvas):
        footer_text = f"Lungwise Risk Screening Report  |  {report_id}  |  Patient: {name}"[:110]

    doc.build(story, canvasmaker=_Canvas)
    return out.getvalue()
