from dataclasses import dataclass

from src.coco.bbox import CocoBBox


@dataclass
class CocoAnnotation:
	id: int
	image_id: int
	category_id: int
	bbox: CocoBBox
	area: float
