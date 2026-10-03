"""Run English-text inference from a trained Neurolab lab bundle."""

import argparse
from pathlib import Path
from neurolab.experiment import predict_texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="artifacts/neurolab-lab")
    parser.add_argument("--text", action="append", help="Repeat for several English texts")
    args = parser.parse_args()
    if not (Path(args.bundle) / "report.json").exists():
        parser.error("Train first: python -m neurolab.experiment --output-dir " + args.bundle)
    texts = args.text or ["I am happy about this result.", "I am worried about tomorrow."]
    print(predict_texts(args.bundle, texts).round(3).to_string(index=False))
    print("V/A/D are estimated text annotation scores on the 1–5 scale.")


if __name__ == "__main__":
    main()
