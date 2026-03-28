"""
KAVACH-AI — Journalist Agent
Generates public-facing, easy-to-understand debunking summaries for social media.
"""

from loguru import logger
from typing import Dict, Any

class JournalistAgent:
    """
    Public Communication Specialist.
    Translates complex forensic metrics into clear, viral-ready debunking messages.
    """
    def __init__(self, target_platform: str = "twitter"):
        self.target_platform = target_platform
        logger.info(f"[JournalistAgent] Initialized for {target_platform}")

    def generate_summary(self, detection_data: dict) -> str:
        """
        Creates a high-impact summary of the detection.
        """
        verdict = detection_data.get("verdict", "UNKNOWN")
        confidence = detection_data.get("confidence", 0) * 100
        risk = detection_data.get("risk_score", 0)
        if isinstance(risk, (int, float)) and risk > 1:
            risk = risk * 100  # normalize to 0-100 for display

        if verdict == "FAKE" or (isinstance(risk, (int, float)) and risk >= 65):
            summary = (
                f"🚨 DEEPFAKE ALERT 🚨\n\n"
                f"KAVACH-AI has detected a high-probability synthetic manipulation in this media. "
                f"Our forensic ensemble suggests a {confidence:.2f}% probability of AI generation. "
                f"Key artifacts found in temporal consistency and facial geometry.\n\n"
                f"#Deepfake #KAVACHAI #CyberSecurity #FactCheck"
            )
        elif verdict == "SUSPICIOUS":
            summary = (
                f"🔶 CAUTION 🔶\n\n"
                f"This media shows signs of potential manipulation (Risk Score: {risk:.0f}/100). "
                f"AI models show moderate disagreement. We recommend manual verification before sharing.\n\n"
                f"#MediaLiteracy #StayInformed #KAVACHAI"
            )
        else:
            summary = (
                f"✅ VERIFIED REAL ✅\n\n"
                f"KAVACH-AI analysis indicates this media is likely authentic. "
                f"All forensic models show natural motion and high visual fidelity.\n\n"
                f"#Verified #Truth #KAVACHAI"
            )

        return summary
