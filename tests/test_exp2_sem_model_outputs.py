"""Offline regression checks for the output-free Exp2 runner."""
import importlib.util
import json
import sys
import contextlib
import pytest
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("sem_mo", ROOT / "scripts/run_exp2_sem_model_outputs.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class OutputFreeTests(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def temporary_path(self, tmp_path):
        self.tmp_path = tmp_path

    def test_transient_windows_file_lock_retried(self):
        operation = unittest.mock.Mock(side_effect=[PermissionError(), PermissionError(), "ok"])
        with patch.object(runner.time, "sleep"):
            self.assertEqual(runner.retry_file_operation(operation), "ok")
        self.assertEqual(operation.call_count, 3)

    def test_replace_retry_and_persistent_lock_fallback_recovery(self):
        import contextlib
        import io
        with contextlib.nullcontext(self.tmp_path) as directory:
            path = Path(directory) / "checkpoint.json"
            original_replace = Path.replace
            calls = []

            def transient_replace(source, target):
                calls.append(1)
                if len(calls) < 3:
                    raise PermissionError()
                return original_replace(source, target)

            with patch.object(Path, "replace", transient_replace), patch.object(runner.time, "sleep"):
                runner.save(path, {"results": [1]})
            self.assertEqual(len(calls), 3)
            self.assertEqual(runner.read_json(path)["results"], [1])
            with patch.object(Path, "replace", side_effect=PermissionError()) as replace, \
                 patch.object(runner.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
                runner.save(path, {"results": [1, 2]})
            self.assertEqual(replace.call_count, 6)
            self.assertEqual(runner.read_json(path)["results"], [1])
            self.assertEqual(runner.recover_json(path, result_checkpoint=True)["results"], [1, 2])
            self.assertEqual(len(list(path.parent.glob("*.fallback-*.json"))), 1)

    def test_usage_recovery_ignores_renamed_tmp_and_does_not_double_count(self):
        import contextlib
        import io
        with contextlib.nullcontext(self.tmp_path) as directory:
            journal = Path(directory)
            path = journal / ("a" * 16 + ".transport.json")
            stats = runner.abl.meter()
            stats["calls"] = 1
            runner.save(path, stats)
            stats["calls"] = 2
            with patch.object(Path, "replace", side_effect=PermissionError()), patch.object(runner.time, "sleep"), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.save(path, stats)
            self.assertEqual(runner.aggregate(journal)["calls"], 2)
            tmp = path.with_suffix(".tmp")
            tmp.unlink()
            self.assertEqual(runner.recover_candidates([tmp, path])["calls"], 1)
            self.assertEqual(runner.aggregate(journal)["calls"], 2)

    def test_ablates_entire_group_without_mutating_source(self):
        payload = {"schema_version": "test", "pair_id": "a" * 16, "features": {"nota_final": 7},
                   "model_outputs": {k: 1 for k in runner.FORBIDDEN}}
        source = SimpleNamespace(pair_id="a" * 16, to_dict=lambda: payload)
        view = runner.abl.DossierView(source, True)
        runner.audit_view(view)
        self.assertIn("model_outputs", payload)
        self.assertEqual(view.payload["features"], payload["features"])

    def test_rejects_nested_engine_proxy(self):
        with self.assertRaises(AssertionError):
            runner.audit_view(SimpleNamespace(payload={"schema_version": "x", "pair_id": "a" * 16,
                                                        "features": {"nested": {"p_cal": .9}}}))

    def test_json_and_csv_cohorts_and_duplicates(self):
        with contextlib.nullcontext(self.tmp_path) as directory:
            path = Path(directory) / "pairs.json"
            path.write_text(json.dumps({"results": [{"pair_id": "a" * 16}]}), encoding="utf-8")
            self.assertEqual(runner.load_ids(path), ["a" * 16])
            path.write_text(json.dumps(["a" * 16, "a" * 16]), encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.load_ids(path)
            csv = path.with_suffix(".csv")
            csv.write_text("pair_id\n" + "b" * 16 + "\n", encoding="utf-8")
            self.assertEqual(runner.load_ids(csv), ["b" * 16])

    def test_usage_aggregation_counts_attempts_and_max_not_sum(self):
        with contextlib.nullcontext(self.tmp_path) as directory:
            for i in range(2):
                stats = runner.abl.meter()
                stats.update(calls=3, input_tokens=100, max_request_input_tokens=100, error_statuses={"429": 1})
                runner.abl.save(Path(directory) / f"{i}.transport.json", stats)
            total = runner.aggregate(Path(directory))
            self.assertEqual(total["calls"], 6)
            self.assertEqual(total["input_tokens"], 200)
            self.assertEqual(total["max_request_input_tokens"], 100)
            self.assertEqual(total["error_statuses"], {"429": 2})

    def test_original_prompts_receive_only_ablated_dossier(self):
        payload = {"schema_version": "test", "pair_id": "a" * 16, "features": {"nota_final": 7},
                   "model_outputs": {"p_cal": .9, "base_choice": "MATCH"}}
        source = SimpleNamespace(pair_id="a" * 16, to_dict=lambda: payload)
        view = runner.abl.DossierView(source, True)
        reviewer = runner.ant.AnthropicReviewer(api_key="offline-not-a-key", model="claude-haiku-5-5")
        for role in ("Agent-A", "Agent-B"):
            messages = reviewer._build_messages(view, agent_role=role)
            self.assertEqual(messages[0]["role"], "system")
            # The unchanged system prompt may mention engine fields; the actual
            # serialized dossier after the final fence must not contain them.
            dossier_text = messages[1]["content"].rsplit("```json\n", 1)[1].split("\n```", 1)[0]
            self.assertEqual(json.loads(dossier_text), view.payload)

    def test_summary_confusion_unsure_and_paired_corrections(self):
        import contextlib
        import io
        import pandas as pd
        ids = [f"{i:016x}" for i in range(5)]
        old_dec = ["NONMATCH", "NONMATCH", "NONMATCH", "MATCH", "UNSURE"]
        new_dec = ["MATCH", "UNSURE", "MATCH", "NONMATCH", "NONMATCH"]
        table = pd.DataFrame({"y": [1, 1, 0, 0, 0], "nota_final": [7] * 5}, index=ids)
        with contextlib.nullcontext(self.tmp_path) as directory:
            root = Path(directory)
            output = root / "data/sprint3b/anthropic_sem_mo"
            for mode in ("vigilancia", "confirmacao"):
                for arm, decisions, folder in (("old", old_dec, "anthropic"), ("new", new_dec, "anthropic_sem_mo")):
                    results = [{"pair_id": pid, "decision": dec, "confidence": .8, "reason_codes": ["INSUFFICIENT_EVIDENCE"],
                                "quality_flags": {"pii_leak_detected": False}} for pid, dec in zip(ids, decisions)]
                    run = {"mode": mode, "model": "claude-haiku-5-5", "results": results, "transport": runner.abl.meter(),
                           "n_errors": 0, "cost_estimate_usd": 0, "elapsed_s": 1, "failed_pair_ids": []}
                    runner.abl.save(root / f"data/sprint3b/{folder}/exp2_llm_sync_{mode}.json", run)
            with patch.object(runner, "ROOT", root), patch.object(runner, "OUT", output), \
                 patch.object(runner, "gold", return_value=table), \
                 patch.dict(runner.os.environ, {"ANTHROPIC_API_KEY": "offline-test-secret-not-persisted"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.summarize()
            summary = pd.read_csv(root / "data/exp2_haiku_sem_mo_resumo.csv")
            row = summary[(summary["mode"] == "vigilancia") & (summary.arm == "new")].iloc[0]
            self.assertEqual([row[k] for k in ("VP", "FP", "FN", "VN", "UNSURE")], [1, 1, 0, 2, 1])
            self.assertEqual(row.fn_old_to_match, 1)
            self.assertEqual(row.fn_old_to_unsure, 1)
            self.assertEqual(row.fp_new_not_old, 1)
            self.assertEqual(row.sens_excl_unsure, 1)
            self.assertEqual(row.sens_unsure_as_nonmatch, .5)
            self.assertAlmostEqual(row.spec_excl_unsure, 2 / 3)
            self.assertEqual(row.mcnemar_old_correct_new_not, 1)
            self.assertEqual(row.mcnemar_old_not_new_correct, 3)
            self.assertEqual(row.mcnemar_exact_p, .625)
            self.assertTrue((root / "data/exp2_haiku_sem_mo_relatorio.md").exists())
            self.assertTrue((output / "auditoria.json").exists())

    def test_smoke_successes_reused_by_full_cohort_resume(self):
        import contextlib
        import io
        ids = [f"{i:016x}" for i in range(5)]
        views = {}
        for pid in ids:
            payload = {"schema_version": "test", "pair_id": pid, "features": {"nota_final": 7}}
            views[pid] = runner.abl.DossierView(SimpleNamespace(pair_id=pid, to_dict=lambda p=payload: p), True)
        seen = []

        def fake_review(pid, view, model, journal):
            seen.append(pid)
            return {"pair_id": pid, "decision": "MATCH", "confidence": .8, "reason_codes": ["INSUFFICIENT_EVIDENCE"],
                    "quality_flags": {"pii_leak_detected": False}, "protocol": "consensus",
                    "agent_a_decision": "MATCH", "agent_b_decision": "MATCH", "total_latency_s": 0, "total_tokens": 0}, None

        with contextlib.nullcontext(self.tmp_path) as directory:
            root = Path(directory)
            cohort = root / "data/sprint3b/anthropic/exp2_llm_sync_vigilancia.json"
            runner.abl.save(cohort, {"pair_ids": ids})
            args = runner.argparse.Namespace(modo="vigilancia", modelo="claude-haiku-5-5", pares=cohort,
                                            saida=root / "data/output.json", workers=4, limit=3)
            with patch.object(runner, "ROOT", root), patch.object(runner, "prepare", return_value=views), \
                 patch.object(runner, "review_pair", side_effect=fake_review), \
                 patch.object(runner.ant, "_extract_json", runner.ant._extract_json), \
                 patch.dict(runner.os.environ, {"ANTHROPIC_API_KEY": "offline-resume-test-key"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                smoke = runner.run(args)
                self.assertEqual(smoke["n_reviewed"], 3)
                self.assertFalse(smoke["complete"])
                args.limit = 0
                full = runner.run(args)
                self.assertEqual(full["n_reviewed"], 5)
                self.assertTrue(full["complete"])
                self.assertEqual(full["failed_pair_ids"], [])
            self.assertEqual(len(seen), 5)
            self.assertEqual(set(seen), set(ids))


if __name__ == "__main__":
    unittest.main()
