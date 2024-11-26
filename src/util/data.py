from pathlib import Path

import torch
from PIL import Image
from torch.utils import data


class ImageDataset(data.Dataset):
	def __init__(self, root: Path, transform=None):
		self.root = root
		self.transform = transform

		extensions = [".jpg", ".jpeg", ".png"]
		self.items = [
			file.name
			for file in root.iterdir()
			if file.suffix.lower() in extensions
		]

	def __len__(self):
		return len(self.items)

	def __getitem__(self, idx):
		try:
			name = self.items[idx]
			path = self.root / name

			original = Image.open(path)
			image = original.convert("RGB")

			if self.transform is not None:
				image = self.transform(image)

			return name, original, image
		except:
			return None


def image_dir_loader(path: Path, batch: int, transform):
	def collate_batch(batch):
		batch = (b for b in batch if b is not None)
		names, originals, images = zip(*batch)
		images = torch.stack(images, 0)
		return names, originals, images

	dataset = ImageDataset(path, transform=transform)
	loader = data.DataLoader(
		dataset,
		batch_size=batch,
		shuffle=False,
		num_workers=4,
		collate_fn=collate_batch,
	)

	return loader
