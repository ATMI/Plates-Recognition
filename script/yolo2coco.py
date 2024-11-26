import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

from PIL import Image
from tqdm import tqdm

from src.box import AnnotatedBox, BoxType
from src.coco.annotation import CocoAnnotation
from src.coco.category import CocoCategory
from src.coco.dataset import CocoDataset, CocoDatasetInfo
from src.coco.image import CocoImage

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
BBOX_LIMIT = 100


@dataclass
class Item:
	image_name: str = None
	image_size: Tuple[int, int] = None
	bboxes: List[AnnotatedBox] = None


def decode_image_size(path: Path):
	try:
		with Image.open(path) as image:
			return image.size
	except Exception:
		print(f"Failed to decode image size: {path}")


def load_bboxes(path: Path, box_cls):
	try:
		with open(path) as file:
			bboxes = [AnnotatedBox.from_str(line, box_cls) for line in file]
			return bboxes
	except Exception:
		print(f"Failed to load bboxes: {path}")


def load_yolo_dataset(labels: Path, images: Path, box_cls):
	items = defaultdict(Item)

	for file in tqdm(labels.iterdir(), "Labels"):
		if file.suffix.lower() != ".txt":
			continue

		bboxes = load_bboxes(file, box_cls)
		if bboxes is None:
			continue

		item = items[file.stem]
		item.bboxes = bboxes

	for file in tqdm(images.iterdir(), "Images"):
		if file.suffix.lower() not in IMAGE_EXTENSIONS:
			continue

		item = items.get(file.stem)
		if item is None:
			continue

		size = decode_image_size(file)
		if size is None:
			continue

		item.image_name = file.name
		item.image_size = size

	items = [
		item
		for item in items.values()
		if item.image_name is not None
	]
	return items


def main(
	src_labels: Path, src_images: Path, src_box: BoxType,
	dst: Path, dst_box: BoxType,
	indent: bool,
):
	dst.mkdir(
		exist_ok=True,
		parents=True,
	)

	src_box_cls = src_box.to_cls()
	dst_box_cls = dst_box.to_cls()

	yolo_dataset = load_yolo_dataset(src_labels, src_images, src_box_cls)
	print(f"Loaded {len(yolo_dataset)} Yolo items")

	coco_images = list()
	coco_annotations = list()
	coco_categories = set()

	for i, item in tqdm(enumerate(yolo_dataset), "Converting"):
		if len(item.bboxes) == 0:
			# print(f"No bboxes found for: {item.image_name}")
			continue

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

		annotations = []
		for j, annotated in enumerate(item.bboxes):
			bbox = annotated.box.to(dst_box_cls, image.width, image.height)
			annotation = CocoAnnotation(
				id=i * BBOX_LIMIT + j,
				image_id=i,
				category_id=annotated.cls,
				bbox=bbox,
				area=bbox.area()
			)
			annotations.append(annotation)
		coco_annotations += annotations

	categories = "0123456789ABEKMHOPCTYX"
	coco_categories = [
		CocoCategory(
			id=category,
			name=categories[category],
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

	print("Saving")
	dataset_path = dst / "dataset.json"
	coco_dataset.save(dataset_path, 2 if indent else None)


if __name__ == "__main__":
	args = argparse.ArgumentParser()
	args.add_argument("src_labels", type=Path)
	args.add_argument("src_images", type=Path)
	args.add_argument("src_box", type=BoxType)
	args.add_argument("dst", type=Path)
	args.add_argument("dst_box", type=BoxType)
	args.add_argument("--indent", type=bool, default=False)

	args = args.parse_args()
	main(args.src_labels, args.src_images, args.src_box, args.dst, args.dst_box, args.indent)
