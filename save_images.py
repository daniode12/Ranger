import config_vgg
from imutils import paths
import cv2
import os

for csvPath in paths.list_files(config_vgg.ANNOTS_PATH, validExts=(".csv")):
	# load the contents of the current CSV annotations file
	rows = open(csvPath).read().strip().split("\n")

	# loop over the rows
	for row in rows:
		# break the row into the filename, bounding box coordinates,
		# and class label
		row = row.split(",")
		#(label, startX, startY, width, height, filename, maxX, maxY) = row
		(filename, startX, startY, endX, endY, label) = row
		imagePath = os.path.sep.join([config_vgg.IMAGES_PATH, filename])
		image = cv2.imread(imagePath)
		writePath = os.path.sep.join([config_vgg.IMAGES_PATH, label, filename])
		cv2.imwrite(writePath, image)