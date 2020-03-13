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
image = read_image(image_name, path_of_images)
original_image = image.copy()

# resize the main image to get the region of interest ROI
list_of_images = os.listdir(path_of_ROI_images)
saved, path = get_roi(original_image)
if saved:
    path = path.split('/')
    image_roi = read_image(path[1] + '.png', path=path[0])

# segmentation of the main image in different fragments
list_of_fragments = os.listdir(path_of_fragments)

image_segmentation(image_roi)
# image_roi = cv2.imread(path_of_fragments+'/0_110_258_55.png')

block_type = ["motion", "looks", "sound", "events", "control", "sensing", "operators", "variables"]
block_name = ["When", "Repeat", "Move", "Wait", "If", "Turn"]

blocktype = block_type[1]
nodes_list = list()
x_coords_list = list()
y_coords_list = list()
child_node = list()
images_fragments_list = list()

root = BlockNode((0, 0, 0, 0), "root", "root", None)

# graph = Graph(root)
for img in list_of_fragments:
    images_fragments_list.append(img)
    image_roi = cv2.imread(path_of_fragments+'/'+img)
    ocr_process(image_roi, east)
    cv2.imshow('roi', image_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
images_fragments_list.sort()

for img, b_type, b_name in zip(images_fragments_list, block_type, block_name):
    img_name = img.split('_')
    x, y, w, h = int(img_name[0]), int(img_name[1]), int(img_name[2]), int(img_name[3].split('.')[0])
    image = read_image(img, path=path_of_fragments)

    node = eval(f'{b_name}({x, y, w, h},"{blocktype}","{b_name}")')

    # graph.add_node(node)
    if x == 0:
        edge = BlockEdge(root, node)
        # graph.add_edge(edge)
    else:
        if nodes_list:
            last_inserted_node = nodes_list.pop()
            x_cord = last_inserted_node.block_coordinates[0]
            y_cord = last_inserted_node.block_coordinates[1]
            # graph.add_node(last_inserted_node)
            if y_cord < y and x_cord < x:
                parent = last_inserted_node
                # graph.add_edge(BlockEdge(last_inserted_node, node))
            elif y_cord < y and x_cord == x:
                # graph.add_edge(BlockEdge(parent, node))
                print()
    nodes_list.append(node)


root = BlockNode((0,0,0,0),"root","root",None)

b0 = BlockH1('Op')
b1 = BlockH1('div')
b3 = BlockH1('neg')
b4 = BlockH1('mult')
b5 = BlockH2('mult',b0)
b6 = BlockH1('mult')
b2 = BlockH2('mod',b3,b4)
# b2.add_blocks(b3,b4)
b1.add_block(b2)

root.add_block_h(b1)
blk1 = control.When((0,0,0,1),"control","when",b1)
blk2 = control.Repeat((0,0,1,0),"control","repeat",b2)
blk3 = control.Move((0,0,1,1),"control","Move",b3)
blk4 = control.Wait((0,1,0,0),"control","wait",b4)
blk5 = control.If((0,1,0,1),"control","if",b5)
blk6 = control.Wait((0,1,1,0),"control","wait",b6)
blk7 = control.Move((0,1,1,1),"control","move",b1)

edge1 = BlockEdge(root,blk1)
edge2 = BlockEdge(root,blk2)
edge3 = BlockEdge(blk2,blk3)
edge4 = BlockEdge(blk2,blk4)
edge5 = BlockEdge(blk2,blk5)
edge6 = BlockEdge(blk5,blk6)
edge7 = BlockEdge(blk5,blk7)

graph = Graph(root)
graph.add_node(blk1)
graph.add_node(blk2)
graph.add_node(blk3)
graph.add_node(blk4)
graph.add_node(blk5)
graph.add_node(blk6)
graph.add_node(blk7)
graph.add_edge(edge1)
graph.add_edge(edge2)
graph.add_edge(edge3)
graph.add_edge(edge4)
graph.add_edge(edge5)
graph.add_edge(edge6)
graph.add_edge(edge7)


# graph.display_graph()
# graph.build_python_file()
# graph.build_graphviz_file()
# graph.show_dfs()