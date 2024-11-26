from pathlib import Path
from typing import List

import torch
from torch import Tensor
from torchvision import transforms

from src.plate.plate import Symbol, Rect
from src.plate.util import image_size_from_args
from src.util import dfine


class PlateRecognizer:
	def __init__(
		self,
		config: Path,
		checkpoint: Path,
		device: torch.device,
		thresh: float,
	):
		super().__init__()
		self.device = device
		self.thresh = thresh

		self.model = dfine.load_torch_model(config, checkpoint)
		self.model.eval()
		self.model.to(device)

	@property
	def transform(self):
		return transforms.Compose([
			transforms.Resize((256, 256)),
			transforms.ToTensor(),
		])

	Symbols = List[Symbol]

	def __extract_symbols__(self, labels: Tensor, boxes: Tensor, scores: Tensor) -> Symbols | None:
		confident = scores > self.thresh
		if not torch.any(confident):
			return None

		# scores = scores[confident]
		boxes = boxes[confident]

		indices = range(len(boxes))
		indices = sorted(indices, key=lambda i: (boxes[i][0], boxes[i][1]))

		symbols = []
		for i in indices:
			box = boxes[i]
			label = labels[i]

			symbol = Symbol(
				id=label.item(),
				rect=Rect(*box.tolist())
			)
			symbols.append(symbol)

		return symbols

	@torch.inference_mode()
	def __call__(self, *args) -> Symbols | List[Symbols] | None:
		image, size = image_size_from_args(self.transform, *args)

		size = size.to(self.device)
		image = image.to(self.device)

		results = self.model(image, size)
		results = zip(*results)

		if image.size(0) == 1:
			labels, boxes, scores = next(results)
			return self.__extract_symbols__(labels, boxes, scores)

		plates = []
		for result in results:
			plate = self.__extract_symbols__(*result)
			plates.append(plate)

		return plates
