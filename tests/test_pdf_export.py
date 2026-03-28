import pytest
import os
from backend.agents.forensic_reporter import ForensicReporterAgent

def test_pdf_generation():
    reporter = ForensicReporterAgent(output_dir="tests/tmp_reports")
    data = {
        "verdict": "FAKE",
        "confidence": 0.98,
        "risk_score": 95,
        "per_model": [
            {"model": "vit_primary", "cal_fake_prob": 0.99, "verdict": "FAKE"},
            {"model": "efficientnet", "cal_fake_prob": 0.97, "verdict": "FAKE"}
        ]
    }
    
    path = reporter.generate_report(data)
    assert os.path.exists(path)
    assert path.endswith(".pdf")
    
    # Cleanup
    if os.path.exists(path):
        os.remove(path)
