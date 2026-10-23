"""Download a pinned model or classify a JSON metadata object from stdin."""
import argparse
import json
import sys

from .artifacts import download_model, model_directory


def main():
    parser = argparse.ArgumentParser(prog="python -m armiger")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="download and verify pinned model data")
    download.add_argument("--directory")
    download.add_argument("--variant", choices=("ultra", "cpu", "all"), default="ultra")
    download.add_argument("--offline", action="store_true")
    classify = commands.add_parser("classify", help="classify one JSON object from stdin")
    classify.add_argument("--model-dir")
    classify.add_argument("--variant", choices=("ultra", "cpu"), default="ultra")
    args = parser.parse_args()
    if args.command == "download":
        print(download_model(args.directory, variant=args.variant, offline=args.offline))
        return
    directory = model_directory(args.model_dir)
    download_model(directory, variant=args.variant, offline=True)
    item = json.load(sys.stdin)
    if args.variant == "cpu":
        from . import ArmigerStandaloneONNX
        result = ArmigerStandaloneONNX(directory / "armiger.int8.onnx", directory / "vocab.json", directory / "calibration.json").predict(item)
    else:
        from . import ArmigerUltraPipeline
        result = ArmigerUltraPipeline.from_directory(directory, preload_catalog=True, rsi_mode="pull_only", opt_in_sharing=False).classify(item).to_dict()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
