import argparse
import json
from pathlib import Path

from .cohort import MANIFEST, build_cohort, read_rows, write_rows


def report_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_cohort(out, rows, audit):
    write_rows(out / "cohort.csv", MANIFEST, rows)
    report_json(out / "cohort_audit.json", audit)
    for split, stats in audit.items():
        print(f"{split:>5}: {stats.get('eligible', 0)}/{stats.get('input_admissions', 0)} eligible, "
              f"{stats.get('risk_positive', 0)} positive")


def training_options(parser, epochs):
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--aux-weight", type=float, default=0.25)
    parser.add_argument("--specificity", type=float, default=0.90)


def main():
    parser = argparse.ArgumentParser(description="Image-centric CXR ward risk research prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    cohort = commands.add_parser("cohort", help="Build and audit the first-ICU/death endpoint")
    cohort.add_argument("--symile", type=Path, required=True)
    cohort.add_argument("--transfers", type=Path, required=True)
    cohort.add_argument("--ward", action="append", default=[])
    cohort.add_argument("--out", type=Path, required=True)
    extract = commands.add_parser("extract", help="Cache CXR, ECG and time-safe lab features")
    for name in ("manifest", "cxr-root", "ecg-root", "labevents", "out"):
        extract.add_argument(f"--{name}", type=Path, required=True)
    extract.add_argument("--device", default="cpu")
    extract.add_argument("--batch-size", type=int, default=16)
    train = commands.add_parser("train", help="Fit and compare modality fusion baselines")
    for name in ("manifest", "features", "out"):
        train.add_argument(f"--{name}", type=Path, required=True)
    training_options(train, 25)
    smoke = commands.add_parser("smoke", help="Run the pipeline on synthetic data")
    smoke.add_argument("--out", type=Path, default=Path("runs/demo"))
    training_options(smoke, 25)
    args = parser.parse_args()

    if args.command == "cohort":
        rows, audit = build_cohort(args.symile, args.transfers, args.ward)
        write_cohort(args.out, rows, audit)
        return
    if args.command == "extract":
        from .features import WEIGHTS, cxr_features, ecg_features, lab_features, save_features

        if args.batch_size < 1 or args.out.suffix != ".npz":
            parser.error("extract requires --batch-size >= 1 and a .npz output path")
        rows = list(read_rows(args.manifest, MANIFEST))
        if not rows:
            parser.error("empty cohort, inspect cohort_audit.json first")
        labs, panels = lab_features(rows, args.labevents)
        ecg = ecg_features(rows, args.ecg_root)
        cxr = cxr_features(rows, args.cxr_root, args.batch_size, args.device, args.out.parent / "weights")
        save_features(args.out, rows, cxr, ecg, labs, WEIGHTS, panels)
        print(f"Features saved for {len(rows)} admissions: {args.out}")
        return

    from .train import run_experiments

    if (args.epochs < 1 or args.bootstrap < 0 or args.seed < 0 or not 0 < args.specificity <= 1
            or not 0 < args.dropout < 1 or not 0 < args.aux_weight < float("inf")):
        parser.error("invalid training options")
    if args.command == "smoke":
        from .demo import demo_data

        rows, audit, features = demo_data(args.out, args.seed)
        write_cohort(args.out, rows, audit)
        args.manifest, args.features = args.out / "cohort.csv", features
    report = run_experiments(args.manifest, args.features, args.epochs, args.seed, args.bootstrap,
                             args.dropout, args.aux_weight, args.specificity)
    report_path = args.out / "metrics.json"
    report_json(report_path, report)
    for name, result in report["models"].items():
        test = result["test"]
        print(f"{name:>18}: AUPRC={test['auprc']:.3f} AUROC={test['auroc']:.3f}")
    print(f"Metrics ({report['source']}): {report_path}")


if __name__ == "__main__":
    main()
