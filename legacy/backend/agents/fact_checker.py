from PIL import Image, ExifTags
import io
import base64
from loguru import logger

class FactCheckerAgent:
    """Extracts EXIF metadata and analyzes origin probability based on technical forensic signals."""

    def check_media_origin(self, scan_result: dict) -> dict:
        """Analyze physical media origin."""
        logger.info(f"[FactChecker] Analyzing origin for scan {scan_result.get('task_id', 'unknown')}")
        
        origin_flags = []
        trust_score = 100
        
        # Analyze available metadata
        if "meta_data" in scan_result and scan_result["meta_data"]:
            meta = scan_result["meta_data"]
            # Example heuristic: if missing software tag or heavily edited
            if not meta.get("Software"):
                origin_flags.append("Missing software signature (potentially scrubbed)")
                trust_score -= 20
        else:
            origin_flags.append("No EXIF metadata found")
            trust_score -= 30
            
        verdict = scan_result.get("verdict", "REAL")
        if verdict == "FAKE":
            origin_flags.append("Confirmed synthetic manipulation by primary models")
            trust_score -= 50
            
        return {
            "origin_flags": origin_flags,
            "trust_score": max(0, trust_score),
            "summary": "Media is highly unverified and lacks provenance." if trust_score < 50 else "Media has standard provenance markers."
        }
