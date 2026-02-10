from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_v3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gzcmd", description="GZ-CMD++ v3 runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run v3 policy pipeline")
    run.add_argument(
        "--input",
        default=str(Path("data") / "COMPARADORSEMIDENT.csv"),
        help="Input CSV path (default: data/COMPARADORSEMIDENT.csv)",
    )
    run.add_argument(
        "--config",
        default=str(Path("gzcmd") / "gzcmd_v3_config.yaml"),
        help="Config YAML path (default: gzcmd/gzcmd_v3_config.yaml)",
    )
    run.add_argument(
        "--mode",
        required=True,
        help="Decision policy mode (e.g., vigilancia, confirmacao)",
    )
    run.add_argument(
        "--output",
        default="",
        help="Output CSV path (if empty, does not write a file)",
    )
    run.add_argument(
        "--no-macd",
        action="store_true",
        help="Disable MACD feature engineering",
    )
    run.add_argument(
        "--llm-used",
        type=int,
        default=0,
        help="Previously used LLM calls in the current window (default: 0)",
    )
    run.add_argument(
        "--p-cal",
        choices=["fit_platt", "load_platt", "stub"],
        default="fit_platt",
        help="How to compute p_cal (default: fit_platt)",
    )
    run.add_argument(
        "--platt-model",
        default="",
        help="Path to a saved Platt model (required for --p-cal load_platt)",
    )
    run.add_argument(
        "--save-platt-model",
        default="",
        help="If set, save fitted Platt model to this path",
    )

    fit = sub.add_parser("fit-calibration", help="Fit and save Platt calibration")
    fit.add_argument(
        "--input",
        default=str(Path("data") / "COMPARADORSEMIDENT.csv"),
        help="Input CSV path (default: data/COMPARADORSEMIDENT.csv)",
    )
    fit.add_argument(
        "--config",
        default=str(Path("gzcmd") / "gzcmd_v3_config.yaml"),
        help="Config YAML path (default: gzcmd/gzcmd_v3_config.yaml)",
    )
    fit.add_argument(
        "--output-model",
        required=True,
        help="Where to write the fitted calibration model (JSON)",
    )
    fit.add_argument(
        "--no-macd",
        action="store_true",
        help="Disable MACD feature engineering (not used by calibration, but keeps pipeline consistent)",
    )

    return p


def _print_summary(summary) -> None:
    print(f"rows={summary.rows}")
    print(f"llm_used={summary.llm_used}/{summary.llm_max}")
    print(f"review_requested={summary.review_requested}")
    print(f"guardrail_counts={summary.guardrails}")
    print(f"action_counts={summary.actions}")
    print(f"p_cal_method={summary.p_cal_method}")
    if summary.p_cal_params is not None:
        print(f"p_cal_params={summary.p_cal_params}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "run":
        out, summary = run_v3(
            input_csv=args.input,
            config_path=args.config,
            mode=args.mode,
            macd_enabled=not args.no_macd,
            llm_used=args.llm_used,
            p_cal=args.p_cal,
            platt_model_path=(args.platt_model or None),
            save_platt_model_path=(args.save_platt_model or None),
        )

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_path, index=False, sep=";")

        _print_summary(summary)
        return 0

    if args.cmd == "fit-calibration":
        # Fit via the runner so we reuse the same loader + cleaning.
        _, summary = run_v3(
            input_csv=args.input,
            config_path=args.config,
            mode="vigilancia",
            macd_enabled=not args.no_macd,
            llm_used=0,
            p_cal="fit_platt",
            save_platt_model_path=args.output_model,
        )
        print(f"wrote_model={args.output_model}")
        print(f"p_cal_params={summary.p_cal_params}")
        return 0

    raise AssertionError(f"Unhandled command: {args.cmd}")
