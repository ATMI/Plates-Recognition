from dataclasses import dataclass


@dataclass
class YoloBBox:
	cls: int
	cx: float
	cy: float
	w: float
	h: float

	@staticmethod
	def from_str(s: str) -> "YoloBBox":
		cls, cx, cy, w, h = s.split(" ")
		return YoloBBox(
			cls=int(cls),
			cx=float(cx),
			cy=float(cy),
			w=float(w),
			h=float(h),
		)
