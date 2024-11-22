from dataclasses import dataclass

from src.yolo.bbox import YoloBBox


@dataclass
class CocoBBox:
	ltx: float
	lty: float
	w: float
	h: float

	@staticmethod
	def from_yolo(bbox: YoloBBox, w: int, h: int) -> "CocoBBox":
		return CocoBBox(
			ltx=w * (bbox.cx - bbox.w / 2),
			lty=h * (bbox.cy - bbox.h / 2),
			w=w * bbox.w,
			h=h * bbox.h,
		)
