from abc import abstractmethod
from copy import copy
from dataclasses import dataclass
from enum import StrEnum
from typing import List


class BoxType(StrEnum):
	LTWH_ABS = "ltwh_abs"
	LTRB_ABS = "ltrb_abs"

	LTWH_REL = "ltwh_rel"
	LTRB_REL = "ltrb_rel"

	def to_cls(self):
		match self:
			case BoxType.LTWH_ABS:
				return LTWHAbsBox
			case BoxType.LTRB_ABS:
				return LTRBAbsBox
			case BoxType.LTWH_REL:
				return LTWHRelBox
			case BoxType.LTRB_REL:
				return LTRBRelBox


@dataclass
class Box:
	@staticmethod
	@abstractmethod
	def from_str(s: str) -> "Box":
		pass

	@staticmethod
	@abstractmethod
	def from_coords(l: List[float]) -> "Box":
		pass

	@staticmethod
	@abstractmethod
	def __from_box__(bbox: "Box") -> "Box":
		pass

	@abstractmethod
	def __str__(self) -> str:
		pass

	def coords(self):
		pass

	@abstractmethod
	def width(self):
		pass

	@abstractmethod
	def height(self):
		pass

	@abstractmethod
	def area(self):
		return self.width() * self.height()

	def to(self, cls, img_w: int, img_h: int):
		box = self
		if isinstance(box, AbsBox) and issubclass(cls, RelBox):
			box = box.to_rel(img_w, img_h)
		elif isinstance(box, RelBox) and issubclass(cls, AbsBox):
			box = box.to_abs(img_w, img_h)
		return cls.__from_box__(box)


@dataclass
class AnnotatedBox:
	cls: int
	box: Box

	@staticmethod
	def from_str(s: str, box_cls) -> "AnnotatedBox":
		cls, s = s.split(" ", maxsplit=1)
		cls = int(cls)
		box = box_cls.from_str(s)
		return AnnotatedBox(cls=cls, box=box)


@dataclass
class RelBox:
	@abstractmethod
	def to_abs(self, img_w: int, img_h: int) -> "AbsBox":
		pass


@dataclass
class AbsBox(Box):
	@abstractmethod
	def to_rel(self, img_w: int, img_h: int) -> "AbsBox":
		pass


@dataclass
class LTWHBox(Box):
	ltx: float
	lty: float
	w: float
	h: float

	@staticmethod
	def from_str(s: str) -> "Box":
		coords = s.split(" ", maxsplit=3)
		coords = (float(coord) for coord in coords)
		return LTWHBox(*coords)

	@staticmethod
	def from_coords(l: List[float]) -> "Box":
		return LTWHBox(*l)

	@staticmethod
	def __from_box__(box: "Box") -> "LTWHBox":
		if isinstance(box, LTWHBox):
			return copy(box)
		if isinstance(box, LTRBBox):
			return LTWHBox(
				ltx=box.ltx,
				lty=box.lty,
				w=box.rbx - box.ltx,
				h=box.rby - box.lty,
			)

	def coords(self):
		return [self.ltx, self.lty, self.w, self.h]

	def width(self):
		return self.w

	def height(self):
		return self.h


@dataclass
class LTRBBox(Box):
	ltx: float
	lty: float
	rbx: float
	rby: float

	@staticmethod
	def from_str(s: str) -> "Box":
		coords = s.split(" ", maxsplit=3)
		coords = (float(coord) for coord in coords)
		return LTRBBox(*coords)

	@staticmethod
	def from_coords(l: List[float]) -> "Box":
		return LTRBBox(*l)

	@staticmethod
	def __from_box__(box: "Box") -> "LTRBBox":
		if isinstance(box, LTWHBox):
			return LTRBBox(
				ltx=box.ltx,
				lty=box.lty,
				rbx=box.ltx + box.w,
				rby=box.lty + box.h,
			)
		if isinstance(box, LTRBBox):
			return copy(box)

	def coords(self):
		return [self.ltx, self.lty, self.rbx, self.rby]

	def width(self):
		return self.rbx - self.ltx

	def height(self):
		return self.rby - self.lty


@dataclass
class LTWHAbsBox(LTWHBox, AbsBox):
	def to_rel(self, img_w: int, img_h: int) -> "LTWHRelBox":
		return LTWHRelBox(
			ltx=self.ltx / img_w,
			lty=self.lty / img_h,
			w=self.w / img_w,
			h=self.h / img_h,
		)


@dataclass
class LTWHRelBox(LTWHBox, RelBox):
	def to_abs(self, img_w: int, img_h: int) -> "LTWHAbsBox":
		return LTWHAbsBox(
			ltx=self.ltx * img_w,
			lty=self.lty * img_h,
			w=self.w * img_w,
			h=self.h * img_h,
		)


@dataclass
class LTRBAbsBox(LTRBBox, AbsBox):
	def to_rel(self, img_w: int, img_h: int) -> "LTRBRelBox":
		return LTRBRelBox(
			ltx=self.ltx / img_w,
			lty=self.lty / img_h,
			rbx=self.rbx / img_w,
			rby=self.rby / img_h,
		)


@dataclass
class LTRBRelBox(LTRBBox, RelBox):
	def to_abs(self, img_w: int, img_h: int) -> "LTRBAbsBox":
		return LTRBAbsBox(
			ltx=self.ltx * img_w,
			lty=self.lty * img_h,
			rbx=self.rbx * img_w,
			rby=self.rby * img_h,
		)
