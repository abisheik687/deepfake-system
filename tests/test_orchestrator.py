"""
DeepShield AI — Automated Test Suite
Tests: normalization, aggregation, API endpoints, pipeline integrity

Run with:
    py -m pytest tests/ -v
    py tests/run_tests.py          ← single script, no pytest needed
"""

import sys
import math
import asyncio
import json
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image, ImageDraw

# ─── Add project root to path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ─── Test 1: Temperature scaling normalization ────────────────────────────────

class TestTemperatureScaling(unittest.TestCase):
    """Unit tests for confidence calibration."""

    def setUp(self):
        from backend.orchestrator.temperature_scaler import calibrate_prob, TEMPERATURES
        self.calibrate = calibrate_prob
        self.TEMPS     = TEMPERATURES

    def test_identity_at_T1(self):
        """T=1.0 should return the original probability."""
        self.TEMPS["test_model"] = 1.0
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            cal = self.calibrate(p, "test_model")
            self.assertAlmostEqual(cal, p, places=4,
                msg=f"T=1 should not change p={p}, got {cal}")

    def test_high_T_pulls_toward_center(self):
        """T > 1 should move extreme probs toward 0.5 (or stay the same for near-integer)."""
        self.TEMPS["overconf_model"] = 2.0
        # Very high fake prob should be pulled toward 0.5 (or stay ≤ original)
        cal_high = self.calibrate(0.98, "overconf_model")
        self.assertLessEqual(cal_high, 0.98, "T>1 should not increase extreme prob")
        self.assertGreater(cal_high, 0.5, "Fake prob should remain > 0.5")

        cal_low = self.calibrate(0.02, "overconf_model")
        self.assertGreaterEqual(cal_low, 0.02, "T>1 should not decrease already-low prob")
        self.assertLess(cal_low, 0.5, "Real prob should remain < 0.5")

    def test_output_in_valid_range(self):
        """Calibrated probability must always be in [0, 1]."""
        for p in [0.001, 0.1, 0.5, 0.9, 0.999]:
            for T in [0.5, 1.0, 1.5, 2.0, 3.0]:
                self.TEMPS["range_test"] = T
                cal = self.calibrate(p, "range_test")
                self.assertGreaterEqual(cal, 0.0, f"p={p} T={T} → {cal} < 0")
                self.assertLessEqual(cal, 1.0, f"p={p} T={T} → {cal} > 1")

    def test_monotone(self):
        """Higher raw fake_prob → higher calibrated prob (monotone)."""
        self.TEMPS["mono_test"] = 1.5
        probs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        cals  = [self.calibrate(p, "mono_test") for p in probs]
        for i in range(len(cals) - 1):
            self.assertLess(cals[i], cals[i+1],
                f"Calibration not monotone: {probs[i]}→{cals[i]}, {probs[i+1]}→{cals[i+1]}")


# ─── Test 2: Ensemble Aggregation ────────────────────────────────────────────

class TestEnsembleAggregation(unittest.TestCase):

    def setUp(self):
        from backend.orchestrator.ensemble_aggregator import aggregate, ModelOutput, EnsembleResult
        self.aggregate   = aggregate
        self.ModelOutput = ModelOutput
        self.EnsembleResult = EnsembleResult

    def _make_outputs(self, entries: list[tuple]) -> list:
        """entries: [(model_name, fake_prob, weight)]"""
        results = []
        for name, fp, w in entries:
            verdict = "FAKE" if fp >= 0.5 else "REAL"
            results.append(self.ModelOutput(
                model_name=name, fake_prob=fp, verdict=verdict,
                confidence=fp if fp >= 0.5 else (1-fp), weight=w,
            ))
        return results

    def test_unanimous_fake(self):
        outputs = self._make_outputs([
            ("model_a", 0.95, 1.0),
            ("model_b", 0.88, 0.8),
            ("model_c", 0.91, 0.9),
        ])
        result = self.aggregate(outputs)
        self.assertEqual(result.verdict, "FAKE")
        self.assertGreaterEqual(result.risk_score, 65)
        self.assertAlmostEqual(result.agreement_score, 1.0, places=1)

    def test_unanimous_real(self):
        outputs = self._make_outputs([
            ("model_a", 0.05, 1.0),
            ("model_b", 0.12, 0.8),
            ("model_c", 0.08, 0.9),
        ])
        result = self.aggregate(outputs)
        self.assertEqual(result.verdict, "REAL")
        self.assertLess(result.risk_score, 30)

    def test_split_vote_suspicious(self):
        outputs = self._make_outputs([
            ("model_a", 0.70, 1.0),   # votes FAKE
            ("model_b", 0.30, 1.0),   # votes REAL
        ])
        result = self.aggregate(outputs)
        # With equal weights and split opinion, should be near 50 → SUSPICIOUS
        self.assertGreaterEqual(result.risk_score, 25)
        self.assertLessEqual(result.risk_score, 75)

    def test_high_weight_dominance(self):
        """A model with 4x weight should dominate the ensemble toward FAKE."""
        outputs = self._make_outputs([
            ("trusted", 0.95, 4.0),   # high weight = FAKE
            ("weak1",   0.10, 1.0),
            ("weak2",   0.15, 1.0),
        ])
        result = self.aggregate(outputs)
        # With 4x weight dominating at 0.95 fake, risk should be ≥ 50 (SUSPICIOUS or FAKE)
        self.assertGreaterEqual(result.risk_score, 50,
            f"High-weight FAKE model (w=4) should push risk ≥ 50, got {result.risk_score}")
        self.assertNotEqual(result.verdict, "REAL",
            "High-weight FAKE vote should not produce REAL verdict")

    def test_all_model_failures(self):
        """When all models fail, verdict should be SUSPICIOUS (neutral)."""
        from backend.orchestrator.ensemble_aggregator import ModelOutput
        outputs = [
            ModelOutput(model_name="m1", fake_prob=0.5, verdict="REAL",
                        confidence=0.0, weight=1.0, failed=True),
            ModelOutput(model_name="m2", fake_prob=0.5, verdict="REAL",
                        confidence=0.0, weight=1.0, timed_out=True),
        ]
        result = self.aggregate(outputs)
        self.assertEqual(result.verdict, "SUSPICIOUS")
        self.assertEqual(result.models_used, 0)
        self.assertEqual(result.reliability_index, 0.0)

    def test_partial_failure_continues(self):
        """If 1/3 models fail, the other 2 should still produce a result."""
        from backend.orchestrator.ensemble_aggregator import ModelOutput
        outputs = [
            ModelOutput(model_name="ok1",   fake_prob=0.92, verdict="FAKE",   confidence=0.92, weight=1.0),
            ModelOutput(model_name="ok2",   fake_prob=0.88, verdict="FAKE",   confidence=0.88, weight=0.9),
            ModelOutput(model_name="fail1", fake_prob=0.5,  verdict="REAL",   confidence=0.0,  weight=1.0, failed=True),
        ]
        result = self.aggregate(outputs)
        self.assertEqual(result.models_used, 2)
        self.assertEqual(result.models_failed, 1)
        self.assertEqual(result.verdict, "FAKE")

    def test_risk_score_range(self):
        """Risk score must always be 0–100."""
        for fp in [0.0, 0.25, 0.5, 0.75, 1.0]:
            outputs = self._make_outputs([("m", fp, 1.0)])
            result  = self.aggregate(outputs)
            self.assertGreaterEqual(result.risk_score, 0)
            self.assertLessEqual(result.risk_score, 100)

    def test_deterministic(self):
        """Same inputs → same outputs every time."""
        outputs = self._make_outputs([
            ("a", 0.82, 1.0), ("b", 0.75, 0.8), ("c", 0.91, 1.2),
        ])
        r1 = self.aggregate(outputs)
        r2 = self.aggregate(outputs)
        self.assertEqual(r1.verdict,    r2.verdict)
        self.assertEqual(r1.risk_score, r2.risk_score)
        self.assertEqual(r1.confidence, r2.confidence)


# ─── Test 3: API endpoint smoke tests (no HTTP needed) ────────────────────────

class TestPipelineIntegrity(unittest.TestCase):
    """
    End-to-end tests using frequency analysis only (always available, no download).
    """

    def _make_base64_image(self, is_fake_pattern: bool = False) -> str:
        """Create a test image. Fake pattern = add checkerboard (frequency artifact)."""
        from PIL import Image, ImageDraw
        import base64, numpy as np
        img = Image.new("RGB", (224, 224), color=(128, 64, 32))
        if is_fake_pattern:
            # Add checkerboard — high-frequency DCT signal like GAN artifacts
            arr = np.array(img)
            for i in range(0, 224, 4):
                for j in range(0, 224, 4):
                    if (i + j) % 8 == 0:
                        arr[i:i+4, j:j+4] = [255, 0, 0]
            img = Image.fromarray(arr.astype(np.uint8))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def test_frequency_models_available(self):
        """DCT and FFT models should always be present and enabled."""
        from backend.orchestrator.model_registry import REGISTRY
        self.assertIn("frequency_dct", REGISTRY)
        self.assertIn("frequency_fft", REGISTRY)
        self.assertTrue(REGISTRY["frequency_dct"].enabled)
        self.assertTrue(REGISTRY["frequency_fft"].enabled)

    def test_cache_roundtrip(self):
        """Cache set/get should return identical data."""
        from backend.orchestrator import cache_service
        key  = "test:" + "x" * 16
        data = {"verdict": "FAKE", "risk_score": 87, "confidence": 0.91}
        cache_service.set(key, data, ttl=10)
        got = cache_service.get(key)
        self.assertIsNotNone(got)
        self.assertEqual(got["verdict"],    data["verdict"])
        self.assertEqual(got["risk_score"], data["risk_score"])
        cache_service.invalidate(key)
        self.assertIsNone(cache_service.get(key))

    def test_frequency_inference_output_range(self):
        """Frequency model fake_prob must be in [0, 1]."""
        from backend.orchestrator.task_runner import _infer_frequency
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (224, 224), color=(100, 150, 200))
        fp, conf = _infer_frequency(img, "dct")
        self.assertGreaterEqual(fp, 0.0)
        self.assertLessEqual(fp, 1.0)
        self.assertGreaterEqual(conf, 0.0)

    def test_model_registry_keys_valid(self):
        """All registry entries must have required fields."""
        from backend.orchestrator.model_registry import REGISTRY, ModelSpec
        for name, spec in REGISTRY.items():
            self.assertEqual(spec.name, name, f"Name mismatch for {name}")
            self.assertGreater(spec.weight, 0, f"Weight must be > 0 for {name}")
            self.assertGreater(spec.timeout_s, 0, f"Timeout must be > 0 for {name}")
            self.assertIn(spec.kind, ["hf_pipeline", "timm", "frequency", "custom"])

    def test_e2e_frequency_pipeline(self):
        """Full pipeline test using frequency models only (no download needed)."""
        import importlib
        b64 = self._make_base64_image(is_fake_pattern=True)
        orch = importlib.import_module("backend.orchestrator.orchestrator")
        result = asyncio.run(
            orch.analyze(
                image_input  = b64,
                models       = ["frequency_dct", "frequency_fft"],
                use_cache    = False,
                detect_faces = False,
                source       = "test",
            )
        )
        # Validate output schema
        for field in ["verdict", "risk_score", "confidence", "agreement_score",
                      "variance", "reliability_index", "models_used", "per_model", "latency_ms"]:
            self.assertIn(field, result, f"Missing field: {field}")
        self.assertIn(result["verdict"], ["FAKE", "SUSPICIOUS", "REAL"])
        self.assertGreaterEqual(result["risk_score"], 0)
        self.assertLessEqual(result["risk_score"],    100)
        self.assertGreater(result["models_used"], 0)


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_all():
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    for cls in [TestTemperatureScaling, TestEnsembleAggregation, TestPipelineIntegrity]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total   = result.testsRun
    failures= len(result.failures)
    errors  = len(result.errors)
    passed  = total - failures - errors

    print(f"\n{'='*50}")
    print(f"DeepShield AI — Test Results")
    print(f"{'='*50}")
    print(f"  Total:   {total}")
    print(f"  ✅ Pass: {passed}")
    print(f"  ❌ Fail: {failures}")
    print(f"  💥 Error:{errors}")
    print(f"{'='*50}")
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
