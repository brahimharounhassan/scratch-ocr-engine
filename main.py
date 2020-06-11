#!/home/unityone/cv/bin/python3
# -*- coding: utf-8 -*-
# Futures
import __future__
# Generic/Built-in
import datetime
import argparse
import os
# opencv required modules
import numpy as np
import cv2
# local modules
from modules.ocrProcessing import *
from modules.imageProcessing import *
from classes.graphPackage import *
from classes.scratchTree import *
from pythonCode.control import *
from pythonCode import control

# authorship information
__authors__ = "Brahim Haroun Hassan , Choueb Mahamat Issa"
__copyright__ = "Copyright 2020, Project Graduation"
__credits__ = ["Amine Lamine"]
__contact__ = ""
__license__ = "unitOne"
__version__ = "0.1"
__maintainer__ = "Bahim Haroun Hassan"
__email__ = "brahimharoun57@yahoo.fr"
__status__ = "Prototype"

# CODE

image_name = "capture3.png"
path_of_images = r"images"
path_of_ROI_images = r"images_roi"
path_of_fragments = r"fragments_found"
path_of_templates = r"images_templates"
list_of_images = os.listdir(path_of_images)

# read the main image
image = read_image(os.path.join(path_of_images,image_name))
original_image = image.copy()

# resize the main image to get the region of interest ROI
list_of_images = os.listdir(path_of_ROI_images)
saved, path = get_roi(original_image)

if saved:
    image_roi = read_image(path + '.png')

# segmentation of the main image in different fragments
fragment_found = image_segmentation(image_roi)
if fragment_found:
    list_of_fragments = os.listdir(path_of_fragments)
    print("-" * 10 + "\timage successfully segmented\t"+ "-"*10)

block_type = ["motion", "looks", "sound", "events", "control", "sensing", "operators", "variables"]
block_name = ["When", "Repeat", "Move", "Wait", "If", "Turn"]

###########################################"
# import the necessary packages
import imutils

# load the tic-tac-toe image and convert it to grayscale
tictac = image_roi
gray = cv2.cvtColor(tictac, cv2.COLOR_BGR2GRAY)

image = cv2.imread("tetris_blocks.png")
def tictoc(gray):
    # find all contours on the tic-tac-toe board
    cnts = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # compute the area of the contour along with the bounding box
        # to compute the aspect ratio
        area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)

        # compute the convex hull of the contour, then use the area of the
        # original contour and the area of the convex hull to compute the
        # solidity
        hull = cv2.convexHull(c)
        hullArea = cv2.contourArea(hull)
        solidity = area / float(hullArea)


        cv2.drawContours(tictac, [c], -1, (0, 0, 255), 2)
        # show the contour properties
        # print("{} (Contour #{}) -- solidity={:.2f}".format(char, i + 1, solidity))

        # show the output image
        cv2.imshow("Output", tictac)
        cv2.waitKey(0)

image_segmentation(image_roi)

def seg(image):

    # load the Tetris block image, convert it to grayscale, and threshold
    # the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]

    # show the original and thresholded images
    cv2.imshow("Original", image)
    cv2.imshow("Thresh", thresh)

    # find external contours in the thresholded image and allocate memory
    # for the convex hull image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    hullImage = np.zeros(gray.shape[:2], dtype="uint8")


    # loop over the contours
    for (i, c) in enumerate(cnts):
        # compute the area of the contour along with the bounding box
        # to compute the aspect ratio
        area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)

        # compute the aspect ratio of the contour, which is simply the width
        # divided by the height of the bounding box
        aspectRatio = w / float(h)

        # use the area of the contour and the bounding box area to compute
        # the extent
        extent = area / float(w * h)

        # compute the convex hull of the contour, then use the area of the
        # original contour and the area of the convex hull to compute the
        # solidity
        hull = cv2.convexHull(c)
        hullArea = cv2.contourArea(hull)
        solidity = area / float(hullArea)

        # visualize the original contours and the convex hull and initialize
        # the name of the shape
        cv2.drawContours(hullImage, [hull], -1, 255, -1)
        cv2.drawContours(image, [c], -1, (240, 0, 159), 3)
        shape = ""
        # if the aspect ratio is approximately one, then the shape is a square
        if aspectRatio >= 0.98 and aspectRatio <= 1.02:
            shape = "SQUARE"

        # if the width is 3x longer than the height, then we have a rectangle
        elif aspectRatio >= 3.0:
            shape = "RECTANGLE"

        # if the extent is sufficiently small, then we have a L-piece
        elif extent < 0.65:
            shape = "L-PIECE"

        # if the solidity is sufficiently large enough, then we have a Z-piece
        elif solidity > 0.80:
            shape = "Z-PIECE"

        # draw the shape name on the image
        cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (240, 0, 159), 2)

        # show the contour properties
        print("Contour #{} -- aspect_ratio={:.2f}, extent={:.2f}, solidity={:.2f}"
              .format(i + 1, aspectRatio, extent, solidity))

        # show the output images
        cv2.imshow("Convex Hull", hullImage)
        cv2.imshow("Image", image)
        cv2.waitKey(0)


# seg(image_roi)


# blocktype = block_type[1]
# nodes_list = list()
# x_coords_list = list()
# y_coords_list = list()
# child_node = list()
# images_fragments_list = list()
#
# root = BlockNode((0, 0, 0, 0), "root", "root", None)
#
# graph = Graph(root)
# for img in list_of_fragments:
#     images_fragments_list.append(img)
#     image_roi = read_image(os.path.join(path_of_fragments,img))
    # ocr_process(image_roi, east)
    # cv2.imshow('roi', image_roi)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# if images_fragments_list :
#     images_fragments_list.sort()
#     for img, b_type, b_name in zip(images_fragments_list, block_type, block_name):
#         img_name = [ int(x.split('.')[0]) if '.' in x else int(x) for x in img.split('_') ]
#         x, y, w, h = int(img_name[0]), int(img_name[1]), int(img_name[2]), int(img_name[3])
#         image = read_image(img, path=path_of_fragments)
#
#         node = eval(f'{b_name}({x, y, w, h},"{blocktype}","{b_name}")')
#     #
#         # graph.add_node(node)
#         if x == 0:
#             edge = BlockEdge(root, node)
#             # graph.add_edge(edge)
#         else:
#             if nodes_list:
#                 last_inserted_node = nodes_list.pop()
#                 x_cord = last_inserted_node.block_coordinates[0]
#                 y_cord = last_inserted_node.block_coordinates[1]
#                 # graph.add_node(last_inserted_node)
#                 if y_cord < y and x_cord < x:
#                     parent = last_inserted_node
#                     # graph.add_edge(BlockEdge(last_inserted_node, node))
#                 elif y_cord < y and x_cord == x:
#                     # graph.add_edge(BlockEdge(parent, node))
#                     print()
#         nodes_list.append(node)
#
#
# root = BlockNode((0,0,0,0),"root","root",None)
#
# b0 = BlockH1('Op')
# b1 = BlockH1('div')
# b3 = BlockH1('neg')
# b4 = BlockH1('mult')
# b5 = BlockH2('mult',b0)
# b6 = BlockH1('mult')
# b2 = BlockH2('mod',b3,b4)
# # b2.add_blocks(b3,b4)
# b1.add_block(b2)
#
# root.add_block_h(b1)
# blk1 = control.When((0,0,0,1),"control","when",b1)
# blk2 = control.Repeat((0,0,1,0),"control","repeat",b2)
# blk3 = control.Move((0,0,1,1),"control","Move",b3)
# blk4 = control.Wait((0,1,0,0),"control","wait",b4)
# blk5 = control.If((0,1,0,1),"control","if",b5)
# blk6 = control.Wait((0,1,1,0),"control","wait",b6)
# blk7 = control.Move((0,1,1,1),"control","move",b1)
#
# edge1 = BlockEdge(root,blk1)
# edge2 = BlockEdge(root,blk2)
# edge3 = BlockEdge(blk2,blk3)
# edge4 = BlockEdge(blk2,blk4)
# edge5 = BlockEdge(blk2,blk5)
# edge6 = BlockEdge(blk5,blk6)
# edge7 = BlockEdge(blk5,blk7)
#
# graph = Graph(root)
# graph.add_node(blk1)
# graph.add_node(blk2)
# graph.add_node(blk3)
# graph.add_node(blk4)
# graph.add_node(blk5)
# graph.add_node(blk6)
# graph.add_node(blk7)
# graph.add_edge(edge1)
# graph.add_edge(edge2)
# graph.add_edge(edge3)
# graph.add_edge(edge4)
# graph.add_edge(edge5)
# graph.add_edge(edge6)
# graph.add_edge(edge7)


# graph.display_graph()
# graph.build_python_file()
# graph.build_graphviz_file()
# graph.show_dfs()

cv2.waitKey(0)
cv2.destroyAllWindows()