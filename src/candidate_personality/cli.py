"""Command-line interface for a complete reproducible experiment."""

import argparse
from pathlib import Path

from .data import generate_data
from .experiment import predict, train


def main():
    parser = argparse.ArgumentParser(description="Synthetic candidate work-style ML laboratory")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Create fictional data")
    generate.add_argument("--output", default="data/synthetic_candidates.csv")
    generate.add_argument("--rows", type=int, default=1200)
    generate.add_argument("--seed", type=int, default=42)
    training = commands.add_parser("train", help="Compare models and evaluate a held-out set")
    training.add_argument("--data", default="data/synthetic_candidates.csv")
    training.add_argument("--output", default="artifacts")
    training.add_argument("--seed", type=int, default=42)
    inference = commands.add_parser("predict", help="Predict fictional rows using a trusted model")
    inference.add_argument("--model", default="artifacts/model.joblib")
    inference.add_argument("--data", required=True)
    inference.add_argument("--output", default="artifacts/predictions.csv")
    args = parser.parse_args()
    try:
        if args.command == "generate":
            frame = generate_data(args.rows, args.seed)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, index=False)
            print(f"Created {len(frame)} synthetic rows: {args.output}")
        elif args.command == "train":
            result = train(args.data, args.output, args.seed)
            print(
                f"Selected {result['selected_model']}; synthetic test accuracy: "
                f"{result['test']['accuracy']:.1%}. Report: {args.output}/REPORT.md"
            )
        else:
            result = predict(args.model, args.data, args.output)
            print(f"Saved {len(result)} synthetic predictions: {args.output}")
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
