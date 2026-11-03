"""Testes focados, com dados sintéticos sem identificadores pessoais."""

import unittest

import pandas as pd

from scripts.estimar_taxas_erro_revisor import (
    FALLBACK_BANDS,
    assign_bands,
    contradictions,
    engine_profile,
    estimate,
    normalize_decision,
    summarize,
    wilson,
)


class ErrorRatesTests(unittest.TestCase):
    def test_wilson_zero_denominator_and_known_interval(self):
        self.assertEqual(wilson(0, 0), (None, None, None))
        rate, low, high = wilson(5, 10)
        self.assertEqual(rate, 0.5)
        self.assertAlmostEqual(low, 0.23659309)
        self.assertAlmostEqual(high, 0.76340691)
        self.assertAlmostEqual(wilson(0, 10)[2], 0.2775328)
        self.assertAlmostEqual(wilson(10, 10)[1], 0.7224672)
        with self.assertRaises(ValueError):
            wilson(2, 1)

    def test_normalization_and_unknown_rejected(self):
        self.assertEqual(normalize_decision(" nonmatch \n"), "NONMATCH")
        for decision in ("MATCH", "UNSURE", "NONMATCH"):
            self.assertEqual(normalize_decision(decision), decision)
        with self.assertRaises(ValueError):
            normalize_decision("ERROR")

    def test_band_boundaries_and_inclusive_max(self):
        definitions = {"low": {"min": 0, "max": 5}, **FALLBACK_BANDS}
        scores = pd.Series([0, 4.999, 5, 6, 7, 8, 9, 1000])
        self.assertEqual(assign_bands(scores, definitions).tolist(),
                         ["low", "low", "grey_low", "grey_mid", "grey_high", "near_high", "high", "high"])
        definitions = {"single": {"min": 9, "max": 10, "inclusive_max": True}}
        self.assertEqual(assign_bands(pd.Series([10]), definitions).tolist(), ["single"])
        with self.assertRaises(ValueError):
            assign_bands(pd.Series([6]), {"a": {"min": 5, "max": 7}, "b": {"min": 6, "max": 8}})

    def test_unsure_stays_in_class_denominators(self):
        part = pd.DataFrame({"y": [0, 0, 1, 1, 1],
                             "decision": ["MATCH", "UNSURE", "NONMATCH", "UNSURE", "MATCH"]})
        row = summarize(part, "vigilancia", "test", "grey_low", False,
                        {"grey_low": {"e_fp": 0.08, "e_fn": 0.12}})
        self.assertEqual(row["n_vinculos"], 3)
        self.assertEqual(row["e_fp"], 0.5)
        self.assertEqual(row["e_fn"], 1 / 3)
        self.assertEqual(row["u"], 2 / 5)
        self.assertTrue(row["n_lt_10"])
        self.assertAlmostEqual(row["e_fp_abs_diff"], 0.42)

    def test_zero_denominator_preserves_na_and_explicit_engine_fallback(self):
        fixed = {"high": {"e_fp": 0.01, "e_fn": 0.03}}
        row = summarize(pd.DataFrame({"y": [1], "decision": ["MATCH"]}),
                        "confirmacao", "test", "high", True, fixed)
        self.assertIsNone(row["e_fp"])
        self.assertIsNone(row["e_fp_ci95_high"])
        profile = engine_profile([row], fixed, True)
        self.assertEqual(profile["error_rates_by_band"]["high"], {"e_fp": 0.01, "e_fn": 0.0})
        self.assertEqual(profile["fallback_fixed"], {"high": ["e_fp"]})
        self.assertTrue(profile["partial"])

    def test_pool_excludes_partial_and_counts_model_decisions(self):
        data = pd.DataFrame([
            {"mode": "vigilancia", "model": model, "partial": partial,
             "band": "grey_low", "y": 0, "decision": decision}
            for model, partial, decision in (("a", False, "MATCH"), ("b", False, "NONMATCH"), ("k3", True, "MATCH"))
        ])
        rows = estimate(data, {"grey_low": {"min": 5, "max": 6}}, {})
        pooled = next(r for r in rows if r["model"] == "pooled" and r["mode"] == "vigilancia")
        self.assertEqual(pooled["n"], 2)
        self.assertEqual(pooled["e_fp"], 0.5)

    def test_contradictions_ignore_unsure_and_split_bands(self):
        data = pd.DataFrame([
            {"pair_id": pid, "model": model, "mode": "vigilancia", "band": band, "y": 1, "decision": decision}
            for pid, band, a, b in (("synthetic-a", "grey_low", "MATCH", "NONMATCH"),
                                    ("synthetic-b", "grey_mid", "NONMATCH", "MATCH"),
                                    ("synthetic-c", "grey_mid", "MATCH", "UNSURE"))
            for model, decision in (("kimi-k2.5", a), ("deepseek-v4.1-flash", b))
        ])
        rows = contradictions(data, FALLBACK_BANDS)
        self.assertEqual(sum(r["n"] for r in rows), 2)
        self.assertEqual(sum(r["n_vinculos"] for r in rows), 2)


if __name__ == "__main__":
    unittest.main()
