import cv2
import numpy as np
import os
import sys
import pytesseract

def morpholyTransform(image):
    kernelSize = 3
    kernel = np.ones((3,3),np.uint8)
    imgBlur = cv2.GaussianBlur(image, (3,3), 3)
    tmp = cv2.GaussianBlur(image,(kernelSize,kernelSize),3)
    morpho1 = cv2.morphologyEx(tmp, cv2.MORPH_ERODE, kernel)
    return cv2.morphologyEx(morpho1, cv2.MORPH_ERODE, kernel)
def medianBlurFilter(image):
    return cv2.medianBlur(image,3)
def blurFilter(image):
    kernelSize = 3
    return cv2.blur(image,(kernelSize, kernelSize))
def GaussianBlurFilter(image):
    kernelSize = 3
    return cv2.GaussianBlur(image,(kernelSize,kernelSize),3)
def thresholdFilter(image):
    return cv2.threshold(image,127,255,cv2.THRESH_TOZERO |  cv2.THRESH_OTSU)[1]
def bilateralFilter(image):
    diameter = 11
    sigmaColor = 17
    sigmaSpace = 17
    return cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
def filter2DFilter(image):
    return cv2.filter2D(image, -1, np.array([[1 for i in range(3)] for j in range(3)], dtype = np.float) / 9)

filterDictionary = {
        "median" : medianBlurFilter,
        "blur" : blurFilter,
        "morpho" : morpholyTransform,
        "thresh" : thresholdFilter,
        "bilateral" : bilateralFilter,
        "gauss" : GaussianBlurFilter,
        "2d" : filter2DFilter             
    }  

def getResizedImage(image,scale,interpol):
    height = image.shape[0]
    width = image.shape[1]
    newHeight = int((height * scale ) / 100)
    newWidth = int((width * scale) / 100)
    return cv2.resize(image,(newHeight,newWidth),interpolation=interpol)       

def read_image(image,path,color = cv2.IMREAD_UNCHANGED):
    path = os.path.join(path, image)
    image = cv2.imread(path,color)
    if image is None:
        print("Impossible to read the image")
        sys.exit()
    return image
def change_image_color(image, color = cv2.COLOR_BGR2GRAY):
    return cv2.cvtColor(image,color)
def get_contour(image):
    contours,hierarchy = get_all_contours(image)
    for c in contours:
        area = cv2.contourArea(c)
        total_area = image.shape[0]*image.shape[1]
        
        if 0.05<float(area/total_area)<0.8:
            return c
def get_all_contours(image, mode = cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE):
    return cv2.findContours(image,mode,method)
def canny_edges(image,threshold1 = 60,threshold2 = 120):
    return cv2.Canny(image,threshold1,threshold2)
def remove_image_background(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = canny_edges(gray_img)
    ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    img_contours = sorted(img_contours, key=cv2.contourArea)
    for i in img_contours:
        if cv2.contourArea(i) > 15000:
            break
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [i],-1, 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)
def image_matching(image,template):
    gray_image = change_image_color(image)
    template = change_image_color(template)
    w, h = template.shape[::-1]
    match_result = cv2.matchTemplate(gray_image, template, cv2.TM_CCORR_NORMED )
    threshold = 0.99
    loc = np.where(match_result >= threshold)
    template_matched = False
    for point in zip(*loc[::-1]):
        if point :
            template_matched = True
            cv2.rectangle(image, point, (point[0] + w, point[1] + h), (0, 0, 255), 1)
    if template_matched:
        print("Template successfully matched")
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Template don't match")

def get_roi(image):
    gray_image = change_image_color(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_image, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Remove dotted lines
    cnts = get_all_contours(thresh)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 5000:
            cv2.drawContours(thresh, [c], -1, (0,0,0), -1)
    # Fill contours
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    close = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    ori = 255 - cv2.morphologyEx(image, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 15000:
            cv2.drawContours(close, [c], -1, (0,0,0), -1)
    # Smooth contours
    close = 255 - close
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, open_kernel, iterations=3)
    # Look for the contours and draw the results
    ROI_number = 0
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    saved = True
    path = ""
    for c in cnts:
        #here the cut of each entity is made
        x,y,w,h = cv2.boundingRect(c)
        image_ROI = image[y-1:y+h+1, x-1:x+w+1]
        if len(image_ROI) > 60:
            image_found = image_ROI
            # image_result = cv2.rectangle(image,(x,y),(x+w,y+h),(36,255,12),2)
            saved = cv2.imwrite('images_roi/ROI_{}.png'.format(ROI_number), image_ROI)
            path = "images_roi/ROI_"+str(ROI_number)
        ROI_number += 1
    return saved,path

def color_detect(image,color):
    image = change_image_color(image,cv2.COLOR_BGR2RGB)
    colors_dictionary = {
    "blue" : [83,126,255],
    "purple" : [174,0,255],
    "magenta" : [225,0,206],
    "yellow" : [255,199,0],
    "orange" : [255,175,0],
    "dark_orange" : [255,136,0],
    "blue_sky" : [73,174,217],
    "malachite" : [0, 228, 95]
    }
    color = colors_dictionary[color]
    thresh = 40

    lower_color = np.array([color[0] - thresh, color[1] - thresh, color[2] - thresh])
    upper_color = np.array([color[0] + thresh, color[1] + thresh, color[2] + thresh])
    mask = cv2.inRange(image,lower_color, upper_color)
    return change_image_color(cv2.bitwise_and(image, image, mask = mask),cv2.COLOR_RGB2BGR)
    
    
    # edged = cv2.Canny(result,30,200)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    # contours, hierarchy = cv2.findContours(closed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(result,contours,-1,(0,255,0),1)
    # cv2.imshow("Result", change_image_color(result,cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # for c in contours:
    #     x,y,w,h = cv2.boundingRect(c)
    #     if h>50 and w>50:
    #     cv2.destroyAllWindows()

def get_fragments(image):
    gray_image = change_image_color(image,cv2.COLOR_BGR2GRAY)
    # for thresh1 in range(20,100):
    edged_image = canny_edges(image)
    # gray_filtered = cv2.bilateralFilter(gray_image,9, 75, 75)

    edged = canny_edges(gray_image)
    original = image.copy()
    	# applying closing function
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closed,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    	# finding_contours
    # cv2.imshow('original image',orig_image)
    # cv2.drawContours(image,contours,-1,(0,0,255),2)
    # for c in contours:
    #     x,y,w,h=cv2.boundingRect(c)
    #     cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)
    #     accuracy=0.03*cv2.arcLength(c,True)
    #     approx=cv2.approxPolyDP(c,accuracy,True)
    #     cv2.drawContours(image,[approx],0,(0,255,0),2)
    # for c in contours:
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    #     cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    idx = 0
    for c in contours:
        image_heigh, image_width = image.shape[:2]
        x,y,w,h = cv2.boundingRect(c)
        if h>60 and w>60:
            idx+=1
            if x == 0 and y == 0 and y + h == image_heigh:
                continue
            else:
                new_img=image[y:y+h,x:x+w]
                
            # cv2.imshow('tralala',image[(image.shape[0]-new_img.shape[0])-:image.shape[1] - new_img.shape[1]])
            coordinates = r''+str(x)+'_'+str(y)+'_'+str(w)+'_'+str(h)+''
            cv2.imwrite("fragments_found/" + coordinates + '.png', new_img)
            cv2.circle(original,(x,y),0,(0,0,255),3)
            font = cv2.FONT_HERSHEY_PLAIN 
            orgine1 = (x,y) 
            orgine2 = (x,y+h) 
            textx = str((x,y))
            texty = str((x,y+h))
            fontScale = 1
            color = (0, 0, 0)
            thickness = 1
            original = cv2.putText(original, textx, orgine1, font, fontScale,color, thickness, cv2.LINE_AA, False) 
            original = cv2.putText(original, texty, orgine2, font, fontScale,color, thickness,cv2.LINE_AA, False)
            cv2.circle(original,(x,y+h),2,(255,0,0),3)
            cv2.imwrite("original.png", original)
