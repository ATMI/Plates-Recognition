import argparse
import re
import shutil

from pathlib import Path
from tqdm import tqdm

from src.plate.plate import Symbol


def main(src: Path, dst: Path):
	if src == dst:
		print("src and dst can not be the same directory")

	dst.mkdir(parents=True, exist_ok=True)
	pattern = re.compile(r"\[([\s\w]+)]")
	correct_count = 0

	for file in tqdm(src.iterdir()):
		if not file.is_file() or not file.suffix == ".txt":
			continue

		ground_truth = pattern.findall(file.stem)[0]
		ground_truth = ground_truth.replace(" ", "")

		try:
			ground_truth = [Symbol.str2id(s) for s in ground_truth]
		except ValueError as e:
			print(e)

		predicted = []
		with file.open("r") as f:
			for line in f:
				cls, _ = line.split(" ", maxsplit=1)
				cls = int(cls)
				predicted.append(cls)

		if ground_truth == predicted:
			path = dst / file.name
			shutil.copy(file, path)
			correct_count += 1

	print("Correct:", correct_count)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("src", type=Path)
	parser.add_argument("dst", type=Path)

	args = parser.parse_args()
	main(args.src, args.dst)
