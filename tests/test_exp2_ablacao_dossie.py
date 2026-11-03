"""Local, offline tests for M4's information boundary and exact test."""
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "ablacao", Path(__file__).resolve().parents[1] / "scripts/run_exp2_ablacao_dossie.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AblationTests(unittest.TestCase):
    def test_view_preserves_control_and_pair_identity(self):
        original = {"schema_version": "test", "pair_id": "abc", "features": {"nota_final": 8},
                    "model_outputs": {"p_cal": .2, "base_choice": "NONMATCH"}}
        dossier = SimpleNamespace(pair_id="abc", to_dict=lambda: original)
        control = module.DossierView(dossier, False)
        ablated = module.DossierView(dossier, True)
        self.assertEqual(control.pair_id, ablated.pair_id)
        self.assertEqual(json.loads(control.to_json()), original)
        self.assertEqual(json.loads(ablated.to_json()), module.ablate(original))

    def test_removes_entire_output_group_without_mutating_control(self):
        original = {"pair_id": "abc", "features": {"nota_final": 8.1},
                    "model_outputs": {"p_cal": .9, "base_choice": "MATCH", "band": "high",
                                      "evr": .2, "guardrail": "MATCH", "guardrail_reason": "decision"}}
        stripped = module.ablate(original)
        self.assertNotIn("model_outputs", stripped)
        self.assertEqual(stripped["features"], original["features"])
        self.assertIn("p_cal", original["model_outputs"])
        stripped["features"]["nota_final"] = 1
        self.assertEqual(original["features"]["nota_final"], 8.1)

    def test_exact_mcnemar(self):
        self.assertEqual(module.exact_mcnemar(0, 0), 1)
        self.assertEqual(module.exact_mcnemar(5, 0), .0625)
        self.assertEqual(module.exact_mcnemar(0, 5), .0625)
        self.assertEqual(module.exact_mcnemar(3, 3), 1)

    def test_safe_result_drops_free_text(self):
        result = SimpleNamespace(to_dict=lambda: {"pair_id": "abc", "decision": "MATCH",
                                                  "raw": "must not be persisted",
                                                  "evidence_summary": {"supports_match": ["free text"]}})
        self.assertEqual(module.safe_result(result), {"pair_id": "abc", "decision": "MATCH"})

    def test_meter_records_status_without_response(self):
        stats = module.meter()
        module.http_error(stats, 400)
        self.assertEqual(stats["http_errors"], 1)
        self.assertEqual(stats["error_statuses"], {"400": 1})

    def test_go_400_stops_without_response_text(self):
        stats = module.meter()
        module._LOCAL.stats = stats
        response = SimpleNamespace(status_code=400, text="untrusted response not to save")
        transport = module.MeteredRequests(SimpleNamespace(post=lambda *a, **kw: response))
        with self.assertRaises(module.FatalAPIError) as exc:
            transport.post("https://example.invalid")
        self.assertEqual(str(exc.exception), "HTTP_400")
        self.assertEqual(stats["calls"], 1)
        self.assertEqual(stats["http_errors"], 1)

    def test_anthropic_usage_counts_cache_and_tokens(self):
        stats = module.meter()
        response = SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=20,
                                                        cache_read_input_tokens=5, cache_creation_input_tokens=2))
        messages = module.MeteredMessages(SimpleNamespace(create=lambda **kw: response), stats)
        self.assertIs(messages.create(model="test"), response)
        self.assertEqual(stats["calls"], 1)
        self.assertEqual(stats["input_tokens"], 10)
        self.assertEqual(stats["output_tokens"], 20)
        self.assertEqual(stats["cache_read_input_tokens"], 5)

    def test_cost_is_estimate_at_documented_short_context_rate(self):
        stats = module.meter()
        stats.update(input_tokens=1_000_000, output_tokens=1_000_000, max_request_input_tokens=5000)
        self.assertAlmostEqual(module.haiku_cost_estimate(stats), .60)
        stats["max_request_input_tokens"] = 100_001
        self.assertIsNone(module.haiku_cost_estimate(stats))


if __name__ == "__main__":
    unittest.main()
