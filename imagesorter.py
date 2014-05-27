

import cv2
import argparse
import os
import glob
import sys
import signal
import itertools as it, glob
import shutil
from common import draw_str

opencv_filetypes = ["png","jpg","jpeg","tiff","bmp","gif","pbm","pgm","ppm","sr","ras","jpe","jp2","tif"]

def signal_handler(signal, frame):
    print 'You pressed Ctrl+C!'
    sys.exit(0)

def print_targets(targetmap):
	for key in targetmap.keys():
		print("%d: %s" % (key, targetmap[key]))

def main():
	signal.signal(signal.SIGINT, signal_handler)
	parser = argparse.ArgumentParser()

	print("=" * 78)
	print("  Image sorter (the fast kind)                     (c) 2014 Joakim Soderberg")
	print("=" * 78)

	try:
		parser.add_argument("--images", metavar = "IMAGES", nargs = "+",
						default = [],
		                help = "The input images that need sorting.")

		parser.add_argument("--skip", metavar = "COUNT",
						type = int, default = 0,
		                help = "Number of images to skip before starting the sort.")

		parser.add_argument("--copy", action = "store_true",
		                help = "Copy instead of moving images.")

		parser.add_argument("--no_overlay", action = "store_true",
						help = "Don't overlay text.")

		parser.add_argument("--targets", metavar = "TARGETS", nargs = "+",
						default = [],
		                help = "A list of paths for the sort targets. "
		                "The first one will have the '1' key as target, "
		                "the next '2' and so on... up to 9 targets.")

		args = parser.parse_args()
	except Exception as ex:
		parser.print_help()
		print("Error: %s" % ex)
		sys.exit(-1)

	if len(args.targets) <= 0:
		parser.print_help()
		print("FAIL: You must specify at least 1 target")
		sys.exit(-1)

	targets = args.targets[:9]

	if len(args.images) <= 0:
		print("No images specified")

	print("Keyboard shortcuts (%s to):" % ("copy" if args.copy else "move"))

	# Create the targets map.
	i = 1
	targetmap = {}

	for t in targets:
		targetmap[i] = t
		i += 1

	# If a directory is specified instead of images
	# then include any images in it.
	img_paths = []
	for img_path in args.images:
		if os.path.isdir(img_path):
			for filetype in opencv_filetypes:
				img_paths += glob.glob(img_path + "/*.%s" % filetype)
		else:
			img_paths.append(img_path)

	img_count = len(img_paths)
	print("Got %d images. Skipping %d" % (img_count, args.skip))

	# Skip images.
	img_paths = img_paths[args.skip:]

	i = args.skip + 1

	for img_path in img_paths:
		try:
			print("=" * 78)
			print_targets(targetmap)
			print("=" * 78)

			print("#%d: %s" % (i, img_path))

			img = cv2.imread(img_path)

			if not args.no_overlay:
				j = 40
				draw_str(img, (20, 20), "Image: %d of %d" % (i, img_count))

				for key in targetmap.keys():
					draw_str(img, (20, j), "%d - %s" % (key, targetmap[key]))
					j += 20

			cv2.imshow("Image sorter", img)

			key = cv2.waitKey(0)

			if (key == ord('q')):
				break

			if (key >= ord("1")) and (key <= ord("9")):
				num = int(unichr(key))

				if targetmap.has_key(num):
					to_dir = targetmap[num]

					if args.copy:
						shutil.copy(img_path, to_dir)
					else:
						shutil.move(img_path, to_dir)

					print("Moved %s to %s" % (img_path, to_dir))
			i += 1

		except Exception as ex:
			print "  ERROR: %s" % ex

if __name__ == "__main__":
	main()
