import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

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

	def save_description(self, file: Path, indent: int | None = None):
		with open(file, "w") as f:
			json.dump(self, f, default=self.json_serializer, indent=indent)
