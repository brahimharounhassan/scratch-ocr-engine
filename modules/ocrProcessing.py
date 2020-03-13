import cv2
from modules.imageProcessing import *
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import pytesseract

# OCR TRAITMENT


dafault_width = 320
default_height = 320
min_confidence = 0.5
east = "dataset/frozen_east_text_detection.pb"

block_name_list = ["move","mov","steps","step","turn","tur","degrees","degree","degre","degr","go","to","glide","glid","point","poin","direction","direct","change","chang","edge","bounce","bounc","set","rotation","style","if","random","position","when","clicked","clicke","click","space","up","arrow","down","left","right","any","key","pressed","presse","wait","second","second","repeat","repet","forever","then","else","until","stop","all","pick","random","and","or","not","mod","round","pen","apple","contains","contain"]
exceptList = ""#{[(\@&!?\"\'%#;|,.$^€¨ùàçaz€`_/)]}"

def setFilter(image,lang):
    filter_list = {"median" : medianBlurFilter,
                   "blur" : blurFilter ,
                   "morpho" : morpholyTransform ,
                   "thresh":thresholding,
                   "bilateral" : bilateralFilter,
                   "gauss" : GaussianBlurFilter,
                   "2d" : filter2DFilter}
    numbreOfLines = 0
    extractedText = ""
    bestFilter = ""
    worldsFound = 0
    image1 = image
    for key in filter_list:
        tmpImage = filter_list[key](image)
        extractedText = get_text(tmpImage,lang)
        save_extracted_text(extractedText)
        dictionnaryLength = len(getDictionnaryText(extractedText))
        if dictionnaryLength > numbreOfLines:
            numbreOfLines = dictionnaryLength
            bestFilter = key
        elif dictionnaryLength == numbreOfLines:
            tempDictionnary = getDictionnaryText(extractedText)
            numbreOfWorlds = 0
            for line in tempDictionnary:
                for world in tempDictionnary[line].split(" "):
                    if world in block_name_list:
                        numbreOfWorlds += 1
            if numbreOfWorlds > worldsFound:
                worldsFound = numbreOfWorlds
                numbreOfLines = dictionnaryLength
                bestFilter = key
    return image,getDictionnaryText(extractedText),bestFilter


def getDictionnaryText(text):
    tempDictionnary = dict()
    with open("OCR_result/tmpExtractedText.txt","r") as myFile:
        numbreOfLines = 0
        for line in myFile:
            line = line.replace('\n', '')
            line = line.split(" ")
            for word in line:
                if word in block_name_list:
                    numbreOfLines += 1
                    line = " ".join(line)
                    for exceptChar in exceptList:
                        if exceptChar in line:
                            line = line.replace(exceptChar,'')
                    tempDictionnary[numbreOfLines] = line
                    break
    return tempDictionnary

def getFinalResult(image,lang,scale = False):
    interpolationDic = {
        "INTER_LANCZOS4" : cv2.INTER_LANCZOS4,
        "INTER_AREA" : cv2.INTER_AREA,
        "INTER_BITS" : cv2.INTER_BITS,
        "INTER_BITS2" : cv2.INTER_BITS2,
        "INTER_CUBIC" : cv2.INTER_CUBIC,
        "INTER_LINEAR" :cv2.INTER_LINEAR,
        "INTER_NEAREST" : cv2.INTER_NEAREST
        }
    bestInterpolation = ""
    oldImage, oldDictionnaryText, oldBestFilter = setFilter(image,lang)
    oldDictionnaryLength = len(oldDictionnaryText)
    if scale:
        while scale >= 20:
            for interpol in interpolationDic:
                newImage = getResizedImage(image,scale,interpolationDic[interpol])
                try:
                    newImage,newDictionnaryText,newBestFilter = setFilter(newImage,lang)
                except Exception as ex:
                    print (ex)
                newDictionnaryLength = len(newDictionnaryText)
                if newDictionnaryLength > oldDictionnaryLength:
                    oldDictionnaryLength = newDictionnaryLength
                    oldDictionnaryText = newDictionnaryText
                    oldBestFilter = newBestFilter
                    bestInterpolation = interpol
                    bestScale = scale
                    oldImage = newImage   
                    bestScale = scale     
            scale -= 10
    return oldImage,oldDictionnaryText, oldBestFilter, bestInterpolation, scale       


def execute_OCR(image,resize = False,lang = 'eng'):
    
    print('--> TRAITEMENT EN COURS <--')
    if len(image.shape) >= 3:
        image = change_image_color(image,cv2.COLOR_BGR2GRAY)
    if resize:
        image,dic,bestFilter,bestInterpol, bestScale = getFinalResult(image,lang,resize)
    else:
        image,dic,bestFilter,bestInterpol, bestScale  = getFinalResult(image,lang)
    for i in dic:
        print(dic[i])
    # print("\nMeilleur filtre : {} ".format(bestFilter))
    # print("Meilleur interpolation : {} \n".format(bestInterpol))
    # print("Meilleur scale : {} \n".format(bestScale))
    print("Nombre de ligne : {} \n".format(len(dic)))

def image_process(image):
    #img = cv2.fastNlMeansDenoisingColored(img, None, 5,5, 7, 21)
	gray = change_image_color(image,cv2.COLOR_BGR2GRAY)
	kernel = np.ones((5,5),np.uint8)
	#gray = cv2.erode(gray,kernel,iterations = 1)
	#gray = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,kernel)
	#gray = cv2.bilateralFilter(gray,11,17,17)
	#closed = cv2.GaussianBlur(gray,(3,3),0)
	thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	filtred = cv2.GaussianBlur(thresh,(3,3),0)
	# closed = cv2.blur(closed,(3,3),0)
	filtred = cv2.bilateralFilter(filtred,11,17,17)
	return filtred

def get_text(image):
    configuration = r'-l eng --oem 3 --psm 12'
    try:
        text = pytesseract.image_to_string(image,config=configuration,nice=2)
    except RuntimeError as timeout_error:
        pass
    return text

def save_extracted_text(text):
    with open("OCR_result/tmpExtractedText.txt", "w") as myFile:
        myFile.write(text)

def read_extracted_text():
    text = ""
    with open("OCR_result/tmpExtractedText.txt","r") as myFile:
        for line in myFile:
            line = line.replace('\n', '')
            text += line
    return text

def ocr_process(image, east, width=dafault_width, height=default_height, min_confidence=min_confidence):
    original_image = image.copy()
    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (width, height)

    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east)

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    first_detected = dict()
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # draw the bounding box on the image
        startY = 0 if startY < 0 else startY
        startX = 0 if startX < 0 else startX
        first_detected[startY] = (startX, endX, endY)
        # cv2.rectangle(original_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    if first_detected:
        height, width = original_image.shape[:2]
        m = min(sorted(first_detected.keys()))
        new_image = original_image[0:first_detected[m][2] * 2, 0:width]
        new_image = image_process(new_image)
        cv2.imshow('filtred',new_image)
        start = time.time()
        save_extracted_text(get_text(new_image))
        text = read_extracted_text()
        end = time.time()
        print("*" * 30)
        print()
        print(text)
        print()
        print("*" * 30)
        print("[INFO] text extract took {:.6f} seconds".format(end - start))

# # show the output image
# cv2.imshow("Text Detection", original_image)
# cv2.waitKey(0)
