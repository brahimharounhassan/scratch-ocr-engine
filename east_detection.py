#!/usr/bin/python
# -*- coding: utf8 -*-
# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import time

east_model = "frozen_east_text_detection.pb"


def short_y_coordinates(y_coordinates,dict_startY_endY):
    y_coordinates.sort()
    new_coords = list()
    old = 0
    i = -1
    last = 0
    for y in y_coordinates : 
        if y - old > 10:
            if y not in new_coords :new_coords.append(y)
            if last > old :
                new_coords[i] = last
            old = y
            i+=1
        else :
            last = y
    
    return new_coords

def get_bloc_boxes(image,startY,endY,pad=3):
    return image[startY - pad:endY + pad , 0:image.shape[1]]

def decode_predictions(scores, geometry, min_confidence):
    	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < min_confidence:
				continue
			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			# endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			# endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			# startX = int(endX - w)
			# startY = int(endY - h)

			offsetX = offsetX + cos * xData1[x] + sin * xData2[x]
			offsetY = offsetY - sin * xData1[x] + cos * xData2[x]

			startX = -cos * w + offsetX
			startY = -cos * h + offsetY
			endX = -sin * h + offsetX
			endY = sin * w + offsetY
			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

def get_east_fragment(image, height = 320, width = 320, min_confidence = 0.5 , padding = 0.0, crop_padding = 8):

    # load the input image and grab the image dimensions
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (height, width)
    rW = origW / float(newW)
    rH = origH / float(newH)
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested in -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    # load the pre-trained EAST text detector
    # print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east_model)

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)

    start = time.time()

    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    end = time.time()
    # show timing information on text prediction
    # print("[INFO] text detection took {:.6f} seconds".format(end - start))
    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry,min_confidence)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    y_coordinates= list()
    dict_startY_endY = dict()
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)
        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))
        # extract the actual padded ROI
        roi = orig[startY:endY, startX:endX]

        # cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        dict_startY_endY[startY] = endY
        y_coordinates.append(startY)


    new_y_coordinates = short_y_coordinates(y_coordinates,dict_startY_endY)
    # if len(new_y_coordinates) <= 2:
    #     print("Not need to be segmented !")
    #     return image
    if len(new_y_coordinates) >= 2:
        print("Segmentation started...",end=".....")
        words_dictionnary = ""
        images_list = list()
        images_dict = dict()

        startY =  new_y_coordinates[0]
        endY = dict_startY_endY[startY]
        roi_image =  get_bloc_boxes(orig,startY,endY,crop_padding)
        print("Segmentation done.")
        return roi_image

    else:
        # print("This frament, not need to be segmented !")
        return orig



    #
    # for startY in new_y_coordinates :
    #     endY = dict_startY_endY[startY]
    #     print()
    #     image = get_bloc_boxes(orig,startY,endY,crop_padding)
    #     cv2.imshow(str(startY), image)
    #     cv2.waitKey(0)
    #     coordinates = str(startY)+"_"+str(endY)
    #
    #     images_dict[coordinates] = image
    #     images_list.append(image)
    # print("Segmentation done.")

    # start_time = time.time()
    # output = print_text(orig, y_coordinates, dict_startY_endY, padding)
    # # print(text_in_black, text_in_white)
    # end_time = time.time()
    # print('Pytesseract extraction took : {:.6f} seconds.'.format(end_time -  start_time))

    # cv2.imshow("Text Detection", orig)
    # cv2.waitKey(0)
    # return images_dict



def get_east_fragments(image, height = 320, width = 320, min_confidence = 0.5 , padding = 0.0, crop_padding = 8):

    # load the input image and grab the image dimensions
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (height, width)
    rW = origW / float(newW)
    rH = origH / float(newH)
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested in -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    # load the pre-trained EAST text detector
    # print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east_model)

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)

    start = time.time()

    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    end = time.time()
    # show timing information on text prediction
    # print("[INFO] text detection took {:.6f} seconds".format(end - start))
    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry,min_confidence)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    y_coordinates= list()
    dict_startY_endY = dict()
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)
        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))
        # extract the actual padded ROI
        roi = orig[startY:endY, startX:endX]

        # cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        dict_startY_endY[startY] = endY
        y_coordinates.append(startY)


    new_y_coordinates = short_y_coordinates(y_coordinates,dict_startY_endY)
    # if len(new_y_coordinates) <= 2:
    #     print("Not need to be segmented !")
    #     return image
    # if len(new_y_coordinates) >= 2:
    #     print("Segmentation started...",end=".....")
    #     words_dictionnary = ""

    #
    #     startY =  new_y_coordinates[0]
    #     endY = dict_startY_endY[startY]
    #     roi_image =  get_bloc_boxes(orig,startY,endY,crop_padding)
    #     print("Segmentation done.")
    #     return roi_image
    #
    # else:
    #     # print("This frament, not need to be segmented !")
    #     return orig



    #images_list = list()
    # images_dict = dict()
    # for startY in new_y_coordinates :
    #     endY = dict_startY_endY[startY]
    #     print()
    #     image = get_bloc_boxes(orig,startY,endY,crop_padding)
    #     cv2.imshow(str(startY), image)
    #     cv2.waitKey(0)
    #     coordinates = str(startY)+"_"+str(endY)
    #
    #     images_dict[coordinates] = image
    #     images_list.append(image)

    for startY in new_y_coordinates:
        endY = dict_startY_endY[startY]
        image = get_bloc_boxes(orig, startY, endY, 8)
        cv2.imshow('image', image)
        cv2.waitKey(0)

        save_path = "test_frag/"
        cv2.imwrite(save_path + 'bloc_{}.jpg'.format(str(startY)), image)

    # return images_dict
