from dataclasses import dataclass


@dataclass
class CocoImage:
	id: int
	file_name: str
	width: int
	height: int
