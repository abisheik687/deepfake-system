import pytest
import unittest.mock as mock
from fastapi.testclient import TestClient

def test_metrics_endpoint_registration():
    """Verify that the /metrics endpoint is registered in the application."""
    # Mocking init_db to avoid DB connection issues during test
    with mock.patch("backend.main.init_db", new_callable=mock.AsyncMock):
        from backend.main import app
        client = TestClient(app)
        # We don't necessarily need to run the app, just check the routes
        routes = [r.path for r in app.routes]
        assert "/metrics" in routes
        print("\n✅ Prometheus Metrics Route Registration Verified")
