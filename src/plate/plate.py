from dataclasses import dataclass
from typing import List

from PIL import ImageFont
from PIL.Image import Image
from PIL.ImageDraw import ImageDraw
from math import sqrt


@dataclass
class Rect:
	ltx: int
	lty: int
	rbx: int
	rby: int

	def height(self) -> float:
		return self.rby - self.lty

	def width(self) -> float:
		return self.rbx - self.ltx

	def area(self) -> float:
		return self.width() * self.height()

	def coords(self) -> tuple[float, float, float, float]:
		return self.ltx, self.lty, self.rbx, self.rby


SYMBOLS_EN = "0123456789ABEKMHOPCTYX"
SYMBOLS_RU = "0123456789АВЕКМНОРСТУХ"


@dataclass
class Symbol:
	id: int
	rect: Rect

	@staticmethod
	def str2id(symbol: str) -> "int":
		symbol = symbol.upper()
		try:
			symbol = SYMBOLS_RU.index(symbol)
		except ValueError:
			try:
				symbol = SYMBOLS_EN.index(symbol)
			except ValueError:
				raise ValueError(f"Unknown symbol {symbol}")
		return symbol

	def __repr__(self) -> str:
		return SYMBOLS_EN[self.id]


@dataclass
class Plate:
	rect: Rect
	symbols: List[Symbol]

	def __repr__(self):
		return "".join(str(symbol) for symbol in self.symbols)


def draw_rect(image: Image | ImageDraw, rect: Rect):
	if isinstance(image, Image):
		image = ImageDraw(image)

	width = sqrt(rect.area()) / 15
	width = max(1, int(width))

	image.rectangle(
		(rect.ltx, rect.lty, rect.rbx, rect.rby),
		outline="red",
		width=width,
	)


def draw_symbols(image: Image | ImageDraw, symbols: List[Symbol]):
	if isinstance(image, Image):
		image = ImageDraw(image)

	for symbol in symbols:
		draw_rect(image, symbol.rect)


def draw_plate(image: Image | ImageDraw, plate: Plate):
	if isinstance(image, Image):
		image = ImageDraw(image)

	draw_rect(image, plate.rect)
	symbols = "".join(str(symbol) for symbol in plate.symbols)

	size = int(0.75 * plate.rect.height())
	pos = (plate.rect.ltx, plate.rect.rby)
	image.text(pos, symbols, fill="red", font=ImageFont.load_default(size))
