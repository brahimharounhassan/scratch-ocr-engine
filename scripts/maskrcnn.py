
from __future__ import division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import glob
import cv2
import time
import math
import matplotlib as plt
import sys
import json
import datetime
import numpy as np
import skimage.draw
import random
import collections
 
# from tensorflow.keras import backend as K
from keras import backend as K


import logging
logging.getLogger('tensorflow').disabled = True

# Import Mask RCNN

from maskrcnnp.mrcnn.config import Config
from maskrcnnp.mrcnn import model as modellib, utils

# Root directory of the project
ROOT_DIR = os.path.abspath("maskrcnnp")

#Scratch root
main_path = os.path.join(ROOT_DIR,"scratch/")

# Path to the dataset (note this is a shared images directory)
dataset_path = os.path.join(main_path, "dataset/images/")

# Path to the models dir
models_dir = os.path.join(main_path,"models/")

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(main_path, "weights/mask_rcnn_coco.h5")
WEIGHTS_PATH = os.path.join(main_path, "weights/mask_rcnn_block_0054.h5")
#WEIGHTS_PATH = os.path.join(main_path, "weights/mask_rcnn_block_00100.h5")


# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(main_path, "logs/")

sys.path.append(ROOT_DIR)  # To find local version of the library

# For Config and Dataset 
sys.path.append(os.path.join(ROOT_DIR, "scratch/"))  # To find local version
import maskrcnnp.scratch.det as det


#  Setup configuration
dataset_name = 'block'
class_names = ['add','ask','broadcast','changeBy','changeEffectBy','changeSizeBy','changeXBy','changeYBy','clear','clearGraphic','createClone','delete','forever','glideTo','glideToXY','goBack','goTo','goToFront','goToXY','hide','hideVariable','ifElse','ifThen','ifOnEdge','move','nextCostume','nextBackdrop','penDown','playSound','pointInDirection','pointTowards','repeat','repeatUntil','replaceItem','resetTimer','say','sayFor','setEffectTo','setTo','setSizeTo','setTimer','setXTo','setYTo','show','showVariable','stamp','stop','switchCostumeTo','switchBackdropTo','think','thinkFor','turnLeft','turnRight','wait','waitUntil','when']

config = det.DetConfig(dataset_name,class_names)

class InferenceConfig(det.DetConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inf_config = InferenceConfig('block', class_names)
inf_model = modellib.MaskRCNN(mode="inference", config=inf_config, model_dir=models_dir)
inf_model.load_weights(WEIGHTS_PATH,by_name=True)

# def detect_instance(class_names, image_test_dir):
def detect_instance(image):

    # det_filenames = sorted(glob.glob(image_test_dir+'/*.png'))
    # for f in det_filenames:
    #     #print("Processing image {}".format(f))
    #
    #     # test_img = plt.imread(f)
    #     test_img = cv2.imread(f)
    fragment_founds = 0

    fragment_dict = dict()

    if image.shape[-1] == 4:
        test_img = image[..., :3]
    else:
        test_img = image
    #plt.imshow(test_img)
    #visualize.display_images([test_img])
    cp = test_img.copy()
    # Included in the results from detect are the found:
    # class_ids,their scores and masks.
    results = inf_model.detect([test_img], verbose=1)[0]

    #print("Objects detected: ", len(results['class_ids']))
    #print(results)
    ROI_number = 0
    list2=[]
    dic_names={}
    bloc_names=[]
    #cv2.imshow("images",test_img[results['rois']])
    cv2.waitKey(0)
    i = 0
    for roi,class_id in zip(results['rois'],results['class_ids']):
       startY,startX,endY,endX = roi
       #print(startX,"--",endX,'--',startY,'--',endY,"=>",config.ALL_CLASS_NAMES[class_id])
       cv2.rectangle(cp,(startX, startY), (endX, endY),(255,100,0),1)
       cv2.putText(cp,str(config.ALL_CLASS_NAMES[class_id]),((startX+endX)//2,(startY+endY)//2),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255),1)
       list=[startY,endY,startX,endX]
      #cv2.imshow(str(config.ALL_CLASS_NAMES[class_id]),test_img[startY:endY,startX:endX])
       list2.append(list)
       dic_names[startY]=config.ALL_CLASS_NAMES[class_id]
       #bloc_names.append(config.ALL_CLASS_NAMES[class_id].capitalize())
    cv2.imshow("image_originale", cp)
    cv2.waitKey(0)

    sorted_dic=sorted(dic_names.items())
    bloc_names=[sorted_dic[i][1].capitalize() for i in range(len(sorted_dic))]

    #list_sorted=sorted(list2, key=lambda x:x[1])
    saved = True
    path = ""

    for roi in list2:
        startY,endY,startX,endX=roi
        image_ROI = test_img[startY:endY,startX:endX]
        coordinates = r'{}_{}_{}_{}'.format(str(startX), str(startY), str(endX), str(endY))
        fragment_dict[startY] = ( [startX,startY,endX, endY], image_ROI)
        fragment_founds += 1
        i += 1
        saved=cv2.imwrite('images_roi/ROI_{}.png'.format(roi), image_ROI)
        string='fragments_found/{}_{}_{}_{}.png'.format(startX,startY,(endX-startX),(endY-startY))
        cv2.imwrite(string,image_ROI)
        path = "images_roi/ROI_"+str(ROI_number)
        ROI_number+=1
# return saved,path,bloc_names
    return  fragment_dict, fragment_founds
# image_test_dir = os.path.join(main_path, "dataset/val/")
# if debugging:

#r = detect_instance(inf_config.ALL_CLASS_NAMES, image_test_dir)





