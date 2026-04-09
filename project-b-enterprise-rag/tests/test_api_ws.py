from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

try:
    _HAS_FASTAPI_TESTCLIENT = importlib.util.find_spec("fastapi.testclient") is not None
except ModuleNotFoundError:
    _HAS_FASTAPI_TESTCLIENT = False

if _HAS_FASTAPI_TESTCLIENT:
    from fastapi.testclient import TestClient


class WebSocketAskTest(unittest.TestCase):
    @unittest.skipUnless(_HAS_FASTAPI_TESTCLIENT, "fastapi.testclient is not installed")
    def test_ws_ask_final_payload(self) -> None:
        from src.api import app

        fake_out = {
            "answer": "Synthetic answer",
            "retrieved": [
                {
                    "text": "retrieved text",
                    "metadata": {"source": "demo.md", "page": 1},
                    "rerank_score": 0.9,
                }
            ],
            "blocked": False,
            "no_context": False,
            "error": False,
            "error_code": None,
        }

        with patch("src.api.run_pipeline", return_value=fake_out):
            client = TestClient(app)
            with client.websocket_connect("/v1/ws/ask") as ws:
                ack = ws.receive_json()
                self.assertEqual(ack["type"], "ack")

                ws.send_json({"question": "What is this?"})
                status1 = ws.receive_json()
                status2 = ws.receive_json()
                final = ws.receive_json()

                self.assertEqual(status1["type"], "status")
                self.assertEqual(status1["stage"], "retrieval_started")
                self.assertEqual(status2["type"], "status")
                self.assertEqual(status2["stage"], "retrieval_finished")
                self.assertEqual(final["type"], "final")
                self.assertEqual(final["answer"], "Synthetic answer")
                self.assertEqual(final["error"], False)

    @unittest.skipUnless(_HAS_FASTAPI_TESTCLIENT, "fastapi.testclient is not installed")
    def test_ws_ask_validation_error(self) -> None:
        from src.api import app

        client = TestClient(app)
        with client.websocket_connect("/v1/ws/ask") as ws:
            _ = ws.receive_json()  # ack
            ws.send_json({"question": "   "})
            err = ws.receive_json()
            self.assertEqual(err["type"], "error")
            self.assertIn("required", err["error"])


if __name__ == "__main__":
    unittest.main()
