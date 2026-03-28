import google.generativeai as genai
import os
from loguru import logger

class JournalistAgent:
    """Generates human-readable intelligence briefings summarizing the forensic investigation using Gemini API."""

    def __init__(self):
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.enabled = True
        else:
            logger.warning("[JournalistAgent] GEMINI_API_KEY missing - generation will be mocked")
            self.enabled = False

    def generate_briefing(self, scan_result: dict) -> str:
        """Generate a 3-paragraph summary of the findings."""
        logger.info(f"[JournalistAgent] Generating briefing for scan {scan_result.get('task_id', 'unknown')}")
        
        verdict = scan_result.get("verdict", "UNKNOWN")
        risk = scan_result.get("risk_score", 0)
        models = scan_result.get("model_breakdown", {})
        
        if not self.enabled:
            return f"Mock Briefing:\n\nThe KAVACH-AI system analyzed the media and concluded a verdict of {verdict} with a risk score of {risk}. The analysis utilized multiple models ({len(models)} modules engaged).\n\nPlease provide a GEMINI_API_KEY to enable full AI-generated journalist briefings."

        prompt = f"""
        Act as a cybersecurity journalist summarizing a deepfake forensic analysis report.
        Write a concise, precise, and professional 3-paragraph briefing.
        
        DATA:
        - Verdict: {verdict}
        - Risk Score: {risk}/1.0
        - Models used: {models}
        
        Paragraph 1: Executive summary of the verdict and risk.
        Paragraph 2: Technical breakdown of what the models found.
        Paragraph 3: Recommended action or conclusion.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"[JournalistAgent] Gemini generation failed: {e}")
            return f"Failed to generate briefing due to API error: {e}"
