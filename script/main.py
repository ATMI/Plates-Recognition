import argparse
from pathlib import Path

from PIL import Image

from script.detect import prepare_detector
from script.recognize import prepare_recognizer
from src.plate.plate import Plate, draw_plate


def main(path: Path, detect_thresh: float, recognize_thresh: float):
	detector = prepare_detector(detect_thresh)
	recognizer = prepare_recognizer(recognize_thresh)

	for file in path.iterdir():
		try:
			image = Image.open(file).convert("RGB")
			plate = detector(image)
			if not plate:
				print("No plate")
				continue

			plate_image = image.crop(plate.coords())
			symbols = recognizer(plate_image)

			plate = Plate(plate, symbols)
			draw_plate(image, plate)

			image.show()
		except Exception as e:
			print(e)
		except KeyboardInterrupt:
			break


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("path", type=Path)
	parser.add_argument("detect_thresh", type=float)
	parser.add_argument("recognize_thresh", type=float)

	args = parser.parse_args()
	main(args.path, args.detect_thresh, args.recognize_thresh)
