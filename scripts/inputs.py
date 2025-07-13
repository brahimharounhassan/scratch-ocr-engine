#!/usr/bin/python
# -*- coding: utf8 -*-
# import the necessary packages

# ROOT_PATH = os.path.abspath("../")
# sys.path.append(ROOT_PATH)

def method_choice():
    choice = None
    while True:
        choice = input()
        try:
            choice = int(choice)
        except:
            print("No character other than an integer can be accepted !")
        if choice != 1 and choice != 2:
            print("You only have two choices, 1 or 2.")
        else:
            return choice

def graph_choice():
    graph = None
    while True:
        print("Do you want to generate a graph ? (Y|N) (y|n) : ")
        graph = input()
        try:
            graph = str(graph)
        except:
            print("Please, put a valide expression !!!")
        if isinstance(graph, str) and len(graph) == 1 and ( graph.lower() == "y" or graph.lower() == "n") :
            return graph
            break
        else:
            print("Please, put a valide expression !!!")

def segmentation_choice():
    welcomeText = """
                                    ******************************************************
                                    **   _____________________________________________  **
                                    **  |                                             | **
                                    **  |      Choose a segmentation method :         | **
                                    **  |                                             | **
                                    **  |----------/ 1 : Contours based   \-----------| **
                                    **  |----------\ 2 : Mask R-CNN based /-----------| **
                                    **  |_____________________________________________| **
                                    **                                                  **
                                    ******************************************************
    """
    print(welcomeText)
    return method_choice()

def start():
    # ==============================================================
    # ==============================================================
    # ==============================================================
    # ============================================================== #

    welcomeText = """
                                    ******************************************************
                                    **   _____________________________________________  **
                                    **  |                                             | **
                                    **  |       WELCOME TO SCRATCH OCR SCANNER        | **
                                    **  |                                             | **
                                    **  |         Choose a processing method          | **
                                    **  |                                             | **
                                    **  |------------/ 1 : Tesseract-ocr  \-----------| **
                                    **  |------------\ 2 : FOTS method    /-----------| **
                                    **  |_____________________________________________| **
                                    **                                                  **
                                    ******************************************************
    """

    print(welcomeText)
    method = None
    segmentation = None
    graph = None
    choice = method_choice()
    if (choice == 1):
        method = "tesseract"
        segmentation = 'opencv' if segmentation_choice() == 1 else "mask"
        graph = True if graph_choice().lower() == "y" else False

    if (choice == 2):
        method = 'fots'
        segmentation = 'opencv' if segmentation_choice() == 1 else "mask"
        graph = True if graph_choice().lower() == "y" else False

    return  (method,segmentation,graph)
