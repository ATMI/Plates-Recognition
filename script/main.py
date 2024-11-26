import argparse
from pathlib import Path

from PIL import Image

from script.detect import prepare_detector
from script.recognize import prepare_recognizer
from src.plate.plate import Plate, draw_plate


def main(path: Path):
	detector = prepare_detector()
	recognizer = prepare_recognizer()

	while True:
		try:
			name = input("> ")
			image = Image.open(path / name).convert("RGB")
			plate = detector(image)
			if not plate:
				print("No plate")
				continue

			plate_image = image.crop(plate.points())
			symbols = recognizer(plate_image)

			plate = Plate(plate, symbols)
			draw_plate(image, plate)
			image.save("plate.png")
			print(plate)
		except Exception as e:
			print(e)
		except KeyboardInterrupt:
			break


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("path", type=Path)

	args = parser.parse_args()
	main(args.path)
