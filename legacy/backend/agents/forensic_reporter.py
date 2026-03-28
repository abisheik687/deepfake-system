"""
KAVACH-AI — Forensic Reporter Agent
Generates production-grade PDF forensic reports with detection metadata, model breakdown, heatmap, chain of custody.
"""

from loguru import logger
import os
import base64
import tempfile
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

from backend.config import settings


class ForensicReporterAgent:
    """Generates real PDF with detection metadata, model breakdown table, embedded heatmap, SHA256 chain of custody."""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or str(settings.EVIDENCE_DIR)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"[ForensicReporter] Output directory: {self.output_dir}")

    def generate_report(
        self,
        detection_result: dict,
        detection_id: str = None,
        heatmap_path: str = None,
        file_hash_sha256: str = None,
    ) -> str:
        """
        Generate PDF evidence report. Saves to ./evidence/{detection_id}.pdf.
        detection_result: dict with verdict, risk_score, confidence, per_model/model_breakdown, etc.
        """
        report_id = detection_id or f"KAVACH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(report_id))[:64]
        file_path = os.path.join(self.output_dir, f"{safe_id}.pdf")

        doc = SimpleDocTemplate(file_path, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        risk = detection_result.get("risk_score", 0)
        if isinstance(risk, (int, float)) and risk > 1:
            risk = risk / 100.0
        risk_pct = f"{float(risk)*100:.1f}%" if isinstance(risk, (int, float)) else str(risk)

        elements.append(Paragraph(f"Forensic Audit Report: {report_id}", styles["Title"]))
        elements.append(Spacer(1, 12))

        data = [
            ["Metric", "Value"],
            ["Detection ID", str(report_id)],
            ["Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")],
            ["Verdict", detection_result.get("verdict", "N/A")],
            ["Risk Score", risk_pct],
            ["Confidence", f"{float(detection_result.get('confidence', 0))*100:.2f}%"],
        ]
        if file_hash_sha256:
            data.append(["SHA256 (analyzed file)", file_hash_sha256[:32] + "..."])
        t = Table(data, colWidths=[120, 280])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 24))

        elements.append(Paragraph("Model breakdown", styles["Heading2"]))
        per_model = detection_result.get("per_model", [])
        model_breakdown = detection_result.get("model_breakdown", {})
        if per_model:
            model_data = [["Model", "Cal. P(Fake)", "Verdict"]]
            for m in per_model:
                cal = m.get("cal_fake_prob", m.get("raw_fake_prob", 0))
                model_data.append([m.get("model", ""), f"{float(cal):.4f}", m.get("verdict", "")])
        elif model_breakdown:
            model_data = [["Model", "P(Fake)", ""]]
            for name, score in model_breakdown.items():
                model_data.append([name, f"{float(score):.4f}", "FAKE" if score >= 0.5 else "REAL"])
        else:
            model_data = [["Model", "N/A", "N/A"]]
        mt = Table(model_data)
        mt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
        elements.append(mt)
        elements.append(Spacer(1, 24))

        heatmap_b64 = detection_result.get("heatmap_b64")
        if heatmap_b64 or heatmap_path:
            elements.append(Paragraph("Visual evidence (GradCAM heatmap)", styles["Heading2"]))
            if heatmap_path and os.path.exists(heatmap_path):
                elements.append(Image(heatmap_path, width=350, height=350))
            elif heatmap_b64:
                try:
                    raw = heatmap_b64.split(",", 1)[-1] if "," in heatmap_b64 else heatmap_b64
                    buf = base64.b64decode(raw)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp.write(buf)
                        tmp.flush()
                        elements.append(Image(tmp.name, width=350, height=350))
                        os.unlink(tmp.name)
                except Exception as e:
                    elements.append(Paragraph(f"[Heatmap unavailable: {e}]", styles["Normal"]))
        elements.append(Spacer(1, 16))
        elements.append(Paragraph("Chain of custody: This report was generated by KAVACH-AI. Hash of analyzed file recorded above.", styles["Normal"]))

        doc.build(elements)
        logger.success(f"[ForensicReporter] Report saved: {file_path}")
        return file_path
