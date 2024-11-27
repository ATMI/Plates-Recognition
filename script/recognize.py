import argparse
import torch

from pathlib import Path
from src.plate.plate import draw_symbols
from src.plate.recognizer import PlateRecognizer
from src.util.dfine import dfine_handle_path

config = Path("model/DFINE/configs/dfine/custom/plate_recognition_n.yml")
checkpoint = Path("model/DFINE/output/plate_recognition_n_7/best_stg1.pth")


def prepare_recognizer(thresh: float) -> PlateRecognizer:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	recognizer = PlateRecognizer(config, checkpoint, device, thresh)
	return recognizer


def prepare_pipeline(src: Path, thresh: float, batch: int):
	recognizer = prepare_recognizer(thresh)

	for name, image, symbols in dfine_handle_path(src, recognizer, batch):
		if symbols is None or len(symbols) == 0:
			# print(f"Could recognize symbols in {name}")
			continue

		yield name, image, symbols


def draw(dst: Path, pipeline):
	for name, image, symbols in pipeline:
		draw_symbols(image, symbols)
		image.save(dst / name)


def label(dst: Path, pipeline):
	for name, image, symbols in pipeline:
		name = Path(name).with_suffix(".txt")

		with open(dst / name, "w") as f:
			for symbol in symbols:
				cls = symbol.id
				ltx, lty, rbx, rby = symbol.rect.coords()
				f.write(f"{cls} {ltx} {lty} {rbx} {rby}\n")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	subparsers = parser.add_subparsers(dest="command")
	draw_parser = subparsers.add_parser("draw")
	label_parser = subparsers.add_parser("label")

	parser.add_argument("src", type=Path)
	parser.add_argument("dst", type=Path)
	parser.add_argument("--batch", type=int, default=16)
	parser.add_argument("--thresh", type=float, default=0.5)

	args = parser.parse_args()
	args.dst.mkdir(exist_ok=True, parents=True)
	pipeline = prepare_pipeline(args.src, args.thresh, args.batch)

	match args.command:
		case "draw":
			draw(args.dst, pipeline)
		case "label":
			label(args.dst, pipeline)
