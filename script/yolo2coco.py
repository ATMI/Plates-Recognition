import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List

from PIL import Image
from pathlib import Path

from src.coco.annotation import CocoAnnotation
from src.coco.bbox import CocoBBox
from src.coco.category import CocoCategory
from src.coco.dataset import CocoDataset, CocoDatasetInfo
from src.coco.image import CocoImage
from src.yolo.bbox import YoloBBox

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif"]
BBOX_LIMIT = 100


@dataclass
class YoloItem:
	image_name: str = None
	image_size: Tuple[int, int] = None
	bboxes: List[YoloBBox] = None

	def is_valid(self) -> bool:
		return (
				self.image_name is not None and
				self.image_size is not None and
				self.bboxes is not None
		)


def decode_image_size(path: Path):
	try:
		with Image.open(path) as image:
			return image.size
	except Exception:
		print(f"Failed to decode image size: {path}")


def load_bboxes(path: Path):
	try:
		with open(path) as file:
			bboxes = [YoloBBox.from_str(line) for line in file]
			return bboxes
	except Exception:
		print(f"Failed to load bboxes: {path}")


def load_yolo_dataset(path: Path):
	items = defaultdict(YoloItem)

	for file in path.iterdir():
		if file.suffix.lower() in IMAGE_EXTENSIONS:
			size = decode_image_size(file)
			if size is None:
				continue

			item = items[file.stem]
			item.image_name = file.name
			item.image_size = size
		else:
			bboxes = load_bboxes(file)
			if bboxes is None:
				continue

			item = items[file.stem]
			item.bboxes = bboxes

	items = [item for item in items.values() if item.is_valid()]
	return items


def main(src: Path, dst: Path, indent: bool):
	dst.mkdir(
		exist_ok=True,
		parents=True,
	)

	yolo_dataset = load_yolo_dataset(src)
	print(f"Loaded {len(yolo_dataset)} Yolo items")

	coco_images = list()
	coco_annotations = list()
	coco_categories = set()

	for i, item in enumerate(yolo_dataset):
		if len(item.bboxes) == 0:
			raise RuntimeError(f"No bboxes found for: {item.image_name}")

		if len(item.bboxes) > BBOX_LIMIT:
			raise RuntimeError(f"Too many bboxes ({len(item.bboxes)}): {item.image_name}")

		categories = (bbox.cls for bbox in item.bboxes)
		coco_categories.update(categories)

		image = CocoImage(
			id=i,
			file_name=item.image_name,
			width=item.image_size[0],
			height=item.image_size[1],
		)
		coco_images.append(image)

		annotations = [
			CocoAnnotation(
				id=i * BBOX_LIMIT + j,
				image_id=i,
				category_id=bbox.cls,
				bbox=CocoBBox.from_yolo(
					bbox=bbox,
					w=item.image_size[0],
					h=item.image_size[1],
				),
				area=bbox.area(),
			)
			for j, bbox in enumerate(item.bboxes)
		]
		coco_annotations += annotations

	coco_categories = [
		CocoCategory(
			id=category,
			name=str(category),
		)
		for category in coco_categories
	]

	coco_dataset = CocoDataset(
		info=CocoDatasetInfo(
			year=2024,
			version="1.0.0",
			description="Description",
		),
		images=coco_images,
		annotations=coco_annotations,
		categories=coco_categories,
	)

	dataset_path = dst / "dataset.json"
	coco_dataset.save(dataset_path, 2 if indent else None)


if __name__ == "__main__":
	args = argparse.ArgumentParser()
	args.add_argument("src", type=Path)
	args.add_argument("dst", type=Path)
	args.add_argument("--indent", type=bool, default=False)

	args = args.parse_args()
	main(args.src, args.dst, args.indent)
