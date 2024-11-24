import time

from PIL import Image
from PIL.Image import Resampling
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Tuple, List

import argparse
import numpy as np
import onnxruntime as ort
import os

INPUT_SIZE = (640, 640)


def load_image(path: Path) -> Tuple[Path, Image, np.ndarray, np.ndarray] | None:
	try:
		image = Image.open(path)
		size = np.array(image.size)

		original = image
		image = image.convert("RGB")
		image = image.resize(INPUT_SIZE, resample=Resampling.BILINEAR)

		image = np.asarray(image, np.float32)
		image = np.clip(image / 255.0, 0, 1)
		image = np.moveaxis(image, 2, 0)

		return path, original, image, size
	except Exception:
		return None


def load_image_worker(path_queue: Queue, image_queue: Queue):
	while True:
		path = path_queue.get()
		if path is None:
			break

		image = load_image(path)
		if image is None:
			continue

		image_queue.put(image)


def load_path_worker(directory: Path, path_queue: Queue):
	for path in directory.iterdir():
		if not path.is_file():
			continue
		path_queue.put(path)


def load_image_pipeline(directory: Path, workers: int, image_queue: Queue):
	path_queue = Queue(2 * workers)
	image_loaders = [
		Thread(
			target=load_image_worker,
			args=(path_queue, image_queue)
		)
		for _ in range(workers)
	]

	for image_loader in image_loaders:
		image_loader.start()
	load_path_worker(directory, path_queue)

	for image_loader in image_loaders:
		image_loader.join()
	image_queue.put(None)


def collate_batch(image_queue: Queue, batch: int):
	images = []
	end = False

	while not end:
		image = image_queue.get()
		if image is None:
			end = True
		else:
			images.append(image)

		if (len(images) == batch) or (end and len(images) > 0):
			path, original, images, sizes = zip(*images)

			images = np.stack(images)
			sizes = np.stack(sizes)

			yield path, original, images, sizes
			images = []


def crop(images: List[Image], boxes, scores, thresh=0.68):
	results = []

	for i, image in enumerate(images):
		score = scores[i]
		confident = score > thresh

		if not np.any(confident):
			continue

		box = boxes[i][confident]
		score = score[confident]

		score = score * np.sqrt(box[:, 2] * box[:, 3])
		max_score = np.argmax(score)

		box = box[max_score]
		image = image.crop((box[0], box[1], box[2], box[3]))
		results.append(image)

	return results


def save(dst: Path, images: List[Tuple[Path, Image]]):
	for path, image in images:
		path = dst / path.name
		image.save(path)


def main(checkpoint: Path, src: Path, dst: Path, batch: int):
	dst.mkdir(parents=True, exist_ok=True)

	providers = ["CUDAExecutionProvider"]
	session = ort.InferenceSession(checkpoint, providers=providers)

	input_image = "images"
	input_size = "orig_target_sizes"

	output_labels = "labels"
	output_scores = "scores"
	output_boxes = "boxes"

	image_queue = Queue(2 * batch)
	workers = min(os.cpu_count(), batch)

	load_pipeline = Thread(target=load_image_pipeline, args=(src, workers, image_queue))
	load_pipeline.start()

	outputs = [
		output_scores,
		output_labels,
		output_boxes,
	]

	for path, original, images, sizes in collate_batch(image_queue, batch):
		inputs = {
			input_image: images,
			input_size: sizes,
		}

		scores, labels, boxes = session.run(outputs, inputs)
		images = crop(original, boxes, scores)
		save(dst, zip(path, images))

	load_pipeline.join()


# print(outputs)

if __name__ == "__main__":
	args = argparse.ArgumentParser()
	args.add_argument("checkpoint", type=Path)
	args.add_argument("src", type=Path)
	args.add_argument("dst", type=Path)
	args.add_argument("batch", type=int)
	args.add_argument("--draw", action="store_true")

	start = time.time()
	args = args.parse_args()
	main(args.checkpoint, args.src, args.dst, args.batch)
	end = time.time()

	print(end - start)
