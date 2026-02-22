
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from backend.reporting.pdf_generator import pdf_gen
from backend.reporting.alerts import alerter

def test_reporting():
    print("Testing Reporting Engine...")
    
    # Mock Data
    task_id = "test_job_999"
    scan_data = {
        "filename": "suspicious_politician.mp4",
        "verdict": "DEEPFAKE DETECTED",
        "confidence": "HIGH",
        "final_score": 0.98,
        "breakdown": {
            "video_spatial": 0.99,
            "audio_spectral": 0.45,
            "temporal_lstm": 0.88
        }
    }
    
    # 1. Generate PDF
    print("1. Generating PDF...")
    pdf_path = pdf_gen.generate_report(task_id, scan_data)
    print(f"✅ PDF Generated: {pdf_path}")
    
    if os.path.exists(pdf_path):
        print(f"   Filesize: {os.path.getsize(pdf_path)} bytes")
    else:
        print("❌ PDF Generation Failed!")
        return

    # 2. Trigger Alert (Mock)
    print("\n2. Sending Alert...")
    alerter.send_alert(
        recipient="admin@kavach.ai",
        subject=f"Deepfake Detected: {task_id}",
        body=f"High confidence deepfake detected in {scan_data['filename']}. Report attached.",
        attachment_path=pdf_path
    )
    print("✅ Alert Logic Executed (Check Mock Output).")

if __name__ == "__main__":
    test_reporting()
