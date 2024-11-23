import argparse
from pathlib import Path

from src.coco.dataset import CocoDataset


def main(path: Path, ratio: float):
	dataset = CocoDataset.load(path)

	train_path = path.parent / "train.json"
	eval_path = path.parent / "eval.json"

	train_dataset, eval_dataset = dataset.split(ratio)
	train_dataset.save(train_path)
	eval_dataset.save(eval_path)


if __name__ == "__main__":
	args = argparse.ArgumentParser()
	args.add_argument("path", type=Path)
	args.add_argument("--ratio", type=float, default=0.8)

	args = args.parse_args()
	main(args.path, args.ratio)
