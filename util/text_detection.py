# USAGE
# python text_detection.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
from PIL import ImageFont, ImageDraw, Image
import time
import cv2
import base64
import time
from google.cloud import vision
import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.measure.entropy import shannon_entropy

net = cv2.dnn.readNet(r"./util/frozen_east_text_detection.pb")
layerNames = ["feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]


def data_uri_to_cv2_img(uri):
	encoded_data = uri.split(',')[1]
	nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	return img

def segmentImage(uri):

	image = data_uri_to_cv2_img(uri)
	path = './img/img' + str(time.time()) + '.jpg'
	cv2.imwrite(path, image)
	words = detect_document(path)
	return words;
	# cv2.imwrite(path, thresh)

	# if words != []:
	# 	print(words)
	# 	# return words

	# 	b,g,r,a = 0,0,0,0
	# 	temp_img = np.full( shape=(1080, 720, 3), fill_value=255, dtype=np.uint8)
	# 	fontpath = "./static/fonts/OpenDyslexic/OpenDyslexic-Bold.otf"     
	# 	font = ImageFont.truetype(fontpath, 32)
	# 	img_pil = Image.fromarray(temp_img)
	# 	draw = ImageDraw.Draw(img_pil)
	# 	draw.text((50, 100),  ' '.join(words), font=font, fill=(b, g, r, a))
	# 	img = np.array(img_pil)
	# 	cv2.imwrite('./img/modified.jpg', img)
		
	
	# print(paragraph)
	# for block in page.blocks:
	# 	print('\nBlock confidence: {}\n'.format(block.confidence))

	# 	for paragraph in block.paragraphs:
	# 		print('Paragraph confidence: {}'.format(
	# 		paragraph.confidence))

	# 		for word in paragraph.words:
	# 			word_text = ''.join([
	# 				symbol.text for symbol in word.symbols
	# 			])
	# 			print('Word text: {} (confidence: {})'.format(
	# 				word_text, word.confidence))

	# orig = image.copy()
	
	# (H, W) = image.shape[:2]

	# # set the new width and height and then determine the ratio in change
	# # for both the width and height
	# width_img = 320
	# height_img = 320
	# (newW, newH) = (width_img, height_img)
	# rW = W / float(newW)
	# rH = H / float(newH)

	# # resize the image and grab the new image dimensions
	# image = cv2.resize(image, (newW, newH))
	# (H, W) = image.shape[:2]

	# # define the two output layer names for the EAST detector model that
	# # we are interested -- the first is the output probabilities and the
	# # second can be used to derive the bounding box coordinates of text


	# # construct a blob from the image and then perform a forward pass of
	# # the model to obtain the two output layer sets
	# blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	# 	(123.68, 116.78, 103.94), swapRB=True, crop=False)
	# start = time.time()
	# net.setInput(blob)
	# (scores, geometry) = net.forward(layerNames)
	# end = time.time()

	# # show timing information on text prediction
	# print("[INFO] text detection took {:.6f} seconds".format(end - start))

	# # grab the number of rows and columns from the scores volume, then
	# # initialize our set of bounding box rectangles and corresponding
	# # confidence scores
	# (numRows, numCols) = scores.shape[2:4]
	# rects = []
	# confidences = []

	# # loop over the number of rows
	# for y in range(0, numRows):
	# 	# extract the scores (probabilities), followed by the geometrical
	# 	# data used to derive potential bounding box coordinates that
	# 	# surround text
	# 	scoresData = scores[0, 0, y]
	# 	xData0 = geometry[0, 0, y]
	# 	xData1 = geometry[0, 1, y]
	# 	xData2 = geometry[0, 2, y]
	# 	xData3 = geometry[0, 3, y]
	# 	anglesData = geometry[0, 4, y]

	# 	# loop over the number of columns
	# 	for x in range(0, numCols):
	# 		# if our score does not have sufficient probability, ignore it
	# 		if scoresData[x] < confidence:
	# 			continue

	# 		# compute the offset factor as our resulting feature maps will
	# 		# be 4x smaller than the input image
	# 		(offsetX, offsetY) = (x * 4.0, y * 4.0)

	# 		# extract the rotation angle for the prediction and then
	# 		# compute the sin and cosine
	# 		angle = anglesData[x]
	# 		cos = np.cos(angle)
	# 		sin = np.sin(angle)

	# 		# use the geometry volume to derive the width and height of
	# 		# the bounding box
	# 		h = xData0[x] + xData2[x]
	# 		w = xData1[x] + xData3[x]

	# 		# compute both the starting and ending (x, y)-coordinates for
	# 		# the text prediction bounding box
	# 		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
	# 		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
	# 		startX = int(endX - w)
	# 		startY = int(endY - h)

	# 		# add the bounding box coordinates and probability score to
	# 		# our respective lists
	# 		rects.append((startX, startY, endX, endY))
	# 		confidences.append(scoresData[x])

	# # apply non-maxima suppression to suppress weak, overlapping bounding
	# # boxes
	# boxes = non_max_suppression(np.array(rects), probs=confidences)

	# # loop over the bounding boxes
	# c = 0
	# images = []
	# imageValues = []
	# images_fp_list = []
	# for (startX, startY, endX, endY) in boxes:
	# 	# scale the bounding box coordinates based on the respective
	# 	# ratios
	# 	startX = int(startX * rW)
	# 	startY = int(startY * rH)
	# 	endX = int(endX * rW)
	# 	endY = int(endY * rH)

	# 	imageValues.append((startX,startY,endX,endY))
	# 	path = './img/img' + str(c) + '.jpg'
	# 	cv2.imwrite(path, orig[startY:endY, startX:endX])
	# 	images_fp_list.append(path)
		
	# 	c += 1
	# 	# For testing
	# 	# images.append(orig[startY:endY, startX:endX]) # Crop from {x, y, w, h } => {0, 0, 300, 400}
	# 	# cv2.imshow("cropped" + str(c), orig[startY:endY, startX:endX])
	# 	# cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

		
	# 	#cv2.waitKey(0)

	# 	# draw the bounding box on the image
	# word_selections = []
	# b,g,r,a = 0,0,0,0

	
	# for image_fp in images_fp_list:
	# 	word = detect_document(image_fp)
	# 	word_selections.append(word)
	# 	print(word)
	# 	# word_selections.append(detect_document(image_fp))

	# for i in range(len(word_selections)):
	# 	# Image Vals => (startX, startY,endX,endY)
	# 	startX = imageValues[i][0]
	# 	startY = imageValues[i][1]
	# 	endX = imageValues[i][2]
	# 	endY = imageValues[i][3]

	# 	temp_img = np.full( shape=(endX-startX, endY-startY, 3), fill_value=255, dtype=np.uint8)
	# 	fontpath = "./static/fonts/OpenDyslexic/OpenDyslexic-Regular.otf"     
	# 	font = ImageFont.truetype(fontpath, 32)
	# 	img_pil = Image.fromarray(temp_img)
	# 	draw = ImageDraw.Draw(img_pil)
	# 	draw.text((50, 100),  word_selections[i], font=font, fill=(b, g, r, a))
	# 	img = np.array(img_pil)	
	# 	orig[startX:endX, startY:endY] = img	

	# show the output image
	# cv2.imwrite('./img/modified.jpg', orig)
	# cv2.imshow("Text Detection", orig)
	# cv2.waitKey(0)

# def detect_document_uri(uri):
# 	"""Detects document features in the file located in Google Cloud
# 	Storage."""
# 	from google.cloud import vision
# 	client = vision.ImageAnnotatorClient()
# 	image = vision.types.Image()
# 	image.source.image_uri = uri

# 	response = client.document_text_detection(image=image)

# 	for page in response.full_text_annotation.pages:
# 		for block in page.blocks:
# 			#print('\nBlock confidence: {}\n'.format(block.confidence))

# 			for paragraph in block.paragraphs:
# 				#   print('Paragraph confidence: {}'.format( paragraph.confidence))
# 				for word in paragraph.words:
# 					word_text = ''.join([symbol.text for symbol in word.symbols])
# 					print('Word text: {} (confidence: {})'.format(word_text, word.confidence))
# 					return word_text
# 					#for symbol in word.symbols:
# 					#     print('\tSymbol: {} (confidence: {})'.format(
# 					#         symbol.text, symbol.confidence))


def detect_document(path):
	"""Detects document features in an image."""
	client = vision.ImageAnnotatorClient()

	with io.open(path, 'rb') as image_file:
		content = image_file.read()

		image = vision.types.Image(content=content)

		response = client.document_text_detection(image=image)

		word_list = []

		for page in response.full_text_annotation.pages:
			for block in page.blocks:
				for paragraph in block.paragraphs:
					for word in paragraph.words:
						word_text = ''.join([
							symbol.text for symbol in word.symbols
						])
						word_list.append(word_text)

		return word_list
