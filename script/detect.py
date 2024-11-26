import argparse
import torch

from pathlib import Path
from src.plate.detector import PlateDetector
from src.plate.plate import draw_rect
from src.util.dfine import dfine_handle_path

config = Path("model/DFINE/configs/dfine/custom/plate_detection_n.yml")
checkpoint = Path("model/DFINE/output/dfine_hgnetv2_n_custom/last.pth")


def prepare_detector(thresh: float) -> PlateDetector:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	detector = PlateDetector(config, checkpoint, device, thresh)
	return detector


def prepare_pipeline(src: Path, thresh: float, batch: int):
	detector = prepare_detector(thresh)

	for name, image, plate in dfine_handle_path(src, detector, batch):
		if plate is None:
			# print(f"Could not find plate in {name}")
			continue

		yield name, image, plate


def draw(dst: Path, pipeline):
	for name, image, plate in pipeline:
		draw_rect(image, plate)
		image.save(dst / name)


def label(dst: Path, pipeline):
	for name, image, plate in pipeline:
		name = Path(name).with_suffix(".txt")
		with open(dst / name, "w") as f:
			ltx, lty, rbx, rby = plate.coords()
			f.write(f"0 {ltx} {lty} {rbx} {rby}\n")


def crop(dst: Path, pipeline):
	for name, image, plate in pipeline:
		image = image.crop(plate.coords())
		image.save(dst / name)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	subparsers = parser.add_subparsers(dest="command")
	draw_parser = subparsers.add_parser("draw")
	crop_parser = subparsers.add_parser("crop")
	label_parser = subparsers.add_parser("label")

	parser.add_argument("src", type=Path)
	parser.add_argument("dst", type=Path)
	parser.add_argument("--batch", type=int, default=16)
	parser.add_argument("--thresh", type=float, default=0.9)

	args = parser.parse_args()
	args.dst.mkdir(exist_ok=True, parents=True)
	pipeline = prepare_pipeline(args.src, args.thresh, args.batch)

	match args.command:
		case "draw":
			draw(args.dst, pipeline)
		case "label":
			label(args.dst, pipeline)
		case "crop":
			crop(args.dst, pipeline)
