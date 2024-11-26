from pathlib import Path

import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from model.DFINE.src.core.yaml_config import YAMLConfig
from src.util.data import image_dir_loader


def load_torch_model(config: str | Path, checkpoint: str | Path):
	if isinstance(config, Path):
		config = str(config)
	if isinstance(checkpoint, Path):
		checkpoint = str(checkpoint)

	cfg = YAMLConfig(cfg_path=config, resume=checkpoint)
	if "HGNetv2" in cfg.yaml_cfg:
		cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

	checkpoint = torch.load(checkpoint, map_location="cpu")
	if "ema" in checkpoint:
		state = checkpoint["ema"]["module"]
	else:
		state = checkpoint["model"]

	# Load train mode state and convert to deploy mode
	cfg.model.load_state_dict(state)

	class Model(nn.Module):
		def __init__(self):
			super().__init__()
			self.model = cfg.model.deploy()
			self.postprocessor = cfg.postprocessor.deploy()

		def forward(self, images, orig_target_sizes):
			outputs = self.model(images)
			outputs = self.postprocessor(outputs, orig_target_sizes)
			return outputs

	model = Model()
	return model


def dfine_handle_path(path: Path, dfine, batch: int):
	if path.is_file():
		image = Image.open(path).convert("RGB")
		result = dfine(image)

		yield path.name, image, result

	elif path.is_dir():
		for names, originals, images in tqdm(image_dir_loader(path, batch, dfine.transform)):
			sizes = torch.stack([torch.tensor(i.size) for i in originals])
			results = dfine(images, sizes)

			for name, original, result in zip(names, originals, results):
				yield name, original, result
