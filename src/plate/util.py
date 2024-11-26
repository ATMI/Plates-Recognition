from typing import List

import torch
from PIL.Image import Image
from torch import Tensor


def image_size_from_args(transform, *args):
	match len(args):
		case 1:
			image = args[0]
			if isinstance(image, Image):
				size = torch.tensor([image.size])
				image = transform(image).unsqueeze(0)
			elif isinstance(image, List):
				size = torch.tensor([i.size for i in image])
				image = transform(image)
			else:
				raise TypeError()
		case 2:
			image, size = args
			if not isinstance(image, Tensor) or not isinstance(size, Tensor):
				raise TypeError()
		case _:
			raise TypeError()
	return image, size
