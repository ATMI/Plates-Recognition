import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from src.coco.annotation import CocoAnnotation
from src.coco.bbox import CocoBBox
from src.coco.category import CocoCategory
from src.coco.image import CocoImage


@dataclass
class CocoDatasetInfo:
	year: int
	version: str
	description: str


@dataclass
class CocoDataset:
	info: CocoDatasetInfo
	images: List[CocoImage]
	annotations: List[CocoAnnotation]
	categories: List[CocoCategory]

	@staticmethod
	def json_serializer(obj):
		if isinstance(obj, CocoBBox):
			return [obj.ltx, obj.lty, obj.w, obj.h]
		return obj.__dict__

	def save(self, file: Path, indent: int | None = None):
		with open(file, "w") as f:
			json.dump(self, f, default=self.json_serializer, indent=indent)

	@staticmethod
	def load(file: Path) -> "CocoDataset":
		with open(file, "r") as f:
			data = json.load(f)

		data["info"] = CocoDatasetInfo(**data["info"])
		data["images"] = [
			CocoImage(**image)
			for image in data["images"]
		]

		for annotation in data["annotations"]:
			ltx, lty, w, h = annotation["bbox"]
			annotation["bbox"] = CocoBBox(ltx, lty, w, h)
		data["annotations"] = [
			CocoAnnotation(**annotation)
			for annotation in data["annotations"]
		]

		data["categories"] = [
			CocoCategory(**category)
			for category in data["categories"]
		]

		dataset = CocoDataset(**data)
		return dataset

	def split(self, ratio: float) -> Tuple["CocoDataset", "CocoDataset"]:
		n = int(len(self.images) * ratio)

		train_images = self.images[:n]
		eval_images = self.images[n:]

		train_ids = set(image.id for image in train_images)
		train_annotations = []
		eval_annotations = []

		for annotation in self.annotations:
			if annotation.image_id in train_ids:
				train_annotations.append(annotation)
			else:
				eval_annotations.append(annotation)

		train_dataset = CocoDataset(
			info=self.info,
			images=train_images,
			annotations=train_annotations,
			categories=self.categories,
		)

		eval_dataset = CocoDataset(
			info=self.info,
			images=eval_images,
			annotations=eval_annotations,
			categories=self.categories,
		)

		return train_dataset, eval_dataset
