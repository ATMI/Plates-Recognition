from pathlib import Path
from typing import List

import torch
from torch import Tensor
from torchvision import transforms

from src.plate.plate import Rect
from src.plate.util import image_size_from_args
from src.util import dfine


class PlateDetector:
	@property
	def transform(self):
		return transforms.Compose([
			transforms.Resize((640, 640)),
			transforms.ToTensor(),
		])

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

	def __extract_plate__(self, boxes: Tensor, scores: Tensor) -> Rect | None:
		confident = scores > self.thresh
		if not torch.any(confident):
			return None

		scores = scores[confident]
		boxes = boxes[confident]

		score = scores * torch.sqrt(boxes[:, 2] * boxes[:, 3])
		max_score = torch.argmax(score)

		box = boxes[max_score]
		box = Rect(*box.tolist())

		return box

	@torch.inference_mode()
	def __call__(self, *args) -> Rect | List[Rect] | None:
		image, size = image_size_from_args(self.transform, *args)

		image = image.to(self.device)
		size = size.to(self.device)

		results = self.model(image, size)
		results = zip(*results)

		if image.size(0) == 1:
			_, boxes, scores = next(results)
			return self.__extract_plate__(boxes, scores)

		plates = []
		for _, boxes, scores in results:
			plate = self.__extract_plate__(boxes, scores)
			plates.append(plate)

		return plates
