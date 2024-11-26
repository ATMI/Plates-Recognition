from dataclasses import dataclass

from src.box import Box


@dataclass
class CocoAnnotation:
	id: int
	image_id: int
	category_id: int
	bbox: Box
	area: float
