#!/usr/bin/python
# -*- coding: utf8 -*-
# import the necessary packages
import pytesseract
import cv2

def get_final_text(text_black, text_white):
    text_black = text_black.strip()
    text_white = text_white.strip()
    numbers = ['0','1','2','3','4','5','6','7','8','9']
    local_dic = ["if", "then", "else", "repeat", "forever", "when", "play", "sound", "until", "done", "move","Move" "steps", "steps", "wait", "second", "seconds", "set", "to", "for", "code", "ask","say","turn","think","show", "glide","change", "point"]
    final_text = ""
    second_finale = ""
    # print("--black--",text_black)
    if text_black :
        text_black = [r for r in text_black.split(' ') if r != '']        
        if text_black[0] in local_dic or text_black[len(text_black)-1] in local_dic:
            final_text = text_black
            if "when" in text_black or "When" in text_black or "clicked" in text_black or "Clicked" in text_black or "else" in text_black or "forever" in text_black:
                return ' '.join([r for r in text_black if r != ''])
        # else :
        #     final_text = text_black
    
    if text_white :
        text_white = [r for r in text_white.split(' ') if r != '']
        if final_text :
            for word in text_white:
                # if word not in local_dic :
                
                # ----------- for REPEAT  --------------
                
                if final_text[0] == "repeat" : 
                    
                    if len(final_text) > 1 :
                        for r in final_text :
                            if r in numbers:
                                final_text[1] = r
                        second_finale = final_text
                    elif word in numbers and len(final_text) == 1:
                        final_text[0] = word
                        second_finale = final_text
                    else :
                        second_finale = final_text                
                
                # ----------- for SAY FOR bloc --------------
                
                elif final_text[0] == "say" and (final_text[len(final_text)-1] == 'second' or final_text[len(final_text)-1] == "seconds"):  
                    if final_text[0] != "say" and text_white[0] == 'say' and word == 'say':
                        final_text.insert(0,word)                  
                    pos_for = None
                    pos_sec = None
                    pos_secs = None
                    for i in range(1,len(final_text)):
                        if final_text[i] == "for":
                            pos_for = i
                        if final_text[i] == 'second':
                            pos_sec = i
                        if final_text[i] == 'seconds':
                            pos_secs = i
                    
                    if pos_for is not None:                        
                        if len(text_white) >= 2:
                            if word in numbers and text_white.index(word) >= 1 and text_white.index(word) <= len(text_white):
                                final_text.insert(len(final_text)-1,word) 
                                text_white[text_white.index(word)] = "" 
                                final_text.insert(1,' '.join(text_white))  
                                second_finale = final_text
                                
                # ----------- for SAY  bloc --------------
                
                elif final_text[0] == "say" and (final_text[len(final_text)-1] != 'second' or final_text[len(final_text)-1] != "seconds")  :
                    if text_white[0] != "say":
                        if len([r for r in final_text if final_text.index(r) > 0]) > len(text_white):
                            second_finale = final_text
                        else:
                            text_white.insert(0,'say')
                            return ' '.join([r for r in text_white if r != ''])
                    # if len(final_text_tmp) > len(text_white_tmp):
                    #     final_text.insert(1,' '.join(final_text[r] for r in range(1,len(final_text))))
                    # else :
                    #     final_text.insert(1,' '.join(text_white[r] for r in range(1,len(text_white))))
                    # second_finale = final_text
                
                # ----------- for MOVE bloc --------------
                
                elif  final_text[0] == "move" or final_text[len(final_text)-1] == "steps" or final_text[len(final_text)-1] == "step":
                    if final_text[0] != "move" and text_white[0] == 'move' and word == 'move':
                        final_text.insert(0,word)
                    if word in numbers :
                        if len(final_text) == 2 :
                            second_finale = final_text.insert(1,word)
                        elif len(final_text) > 2:
                            for r in range(1,len(final_text)-1):
                                final_text[r] = ""
                            second_finale = final_text.insert(1,word)
                    else : 
                        second_finale = final_text
                
                # ----------- for WAIT bloc --------------
                
                elif  final_text[0] == "wait" or final_text[len(final_text)-1] == "second" or final_text[len(final_text)-1] == "seconds":
                    if final_text[0] != "wait" and text_white[0] == 'wait' and word == 'wait':
                        final_text.insert(0,word)
                    if word in numbers :
                        if len(final_text) == 2 :
                            second_finale = final_text.insert(1,word)
                        elif len(final_text) > 2:
                            for r in range(1,len(final_text)-1):
                                final_text[r] = ""
                            final_text.insert(1,word)
                            second_finale = final_text
                    else : 
                        second_finale = final_text
                
                # ----------- for PLAY SOUND bloc --------------
                
                elif final_text[0] == "play" or final_text[0] == "playsound" or final_text[0] == "Play"  or final_text[len(final_text)-1] == "until" or final_text[len(final_text)-1] == "done" :
                    if final_text[0] != "play" and text_white[0] == 'play' and word == 'play':
                        final_text.insert(0,word)
                    if len(final_text) > len(text_white):
                        second_finale = final_text
                    else:
                        second_finale = text_white
                
                # ----------- for SET To bloc --------------

                elif final_text[0] == "set" or ("myvariable" in final_text) or ("my variable" in final_text) :
                    if final_text[0] != "set" and text_white[0] == 'set' and word == 'set':
                        final_text.insert(0,word)
                    if "to" in final_text :
                        pos_final_text = final_text.index("to")
                    if 'to' in text_white:
                        pos_text_white = text_white.index("to")
                    if pos_final_text and pos_text_white :
                        final_text_tmp = [final_text[r] for r in range(pos_final_text,len(final_text))]
                        text_white_tmp = [text_white[r] for r in range(pos_text_white,len(text_white))]
                        if len(final_text_tmp) > len(text_white_tmp):
                            second_finale = final_text
                
                # ----------- for IF THEN bloc --------------
                
                elif final_text[0] == "if" or final_text[len(final_text)-1] == "then":
                    if final_text[0] != "if" and text_white[0] == 'if' and word == 'if':
                        final_text.insert(0,word)
                    pos = None
                    for r in range(0,len(final_text)):
                        if final_text[r] == 'then':
                            pos = r
                    if pos is not None:
                        if final_text[0] =='if':
                            final_text_tmp = [final_text[r] for r in range(1,pos-1)]
                            if len(final_text_tmp) > len(text_white):
                                final_text = ['' for r in range(1,pos)]
                                final_text.insert(1,text_white) 
                                return ' '.join([r for r in final_text if r != ''])
                            else:
                                second_finale = final_text
                            # print('///////////////////////',text_white,pos)
                    
                    # text_white_tmp = [text_white[r] for r in range(1,len(text_white)-1)]
                    # if len(final_text_tmp) > len(text_white_tmp):
                    #     second_finale = final_text
                
                # ----------- for ASK bloc --------------
                
                elif final_text[0] == "ask" or final_text[len(final_text)-1] == "wait" or final_text[len(final_text)-1] == "and" or final_text[len(final_text)-1] == "andwait":
                    if final_text[0] != "ask" and text_white[0] == 'ask' and word == 'ask':
                        final_text.insert(0,word)
                    pos = None
                    for i in range(1,len(final_text)):
                        if final_text[i] == "wait":
                            pos = i
                    if pos is not None :
                        for r in range(1,pos-1):
                            final_text[r] = ''
                        final_text.insert(1,' '.join(text_white))
                        second_finale = final_text
                    # if len(final_text_tmp) > len(text_white_tmp):
                    #     second_finale = final_text
                
                else :
                    pass   
                        # return text_white
                        # print("---final_text---",final_text)
                        # print("---text_white---",text_white)
                        
                        # if pos_final_text :
                        #     if final_text[pos_final_text+1] in numbers:
                        #         second_finale = final_text
                        # if pos_text_white :
                        #     if word in numbers and word in [r for r in range(pos_text_white+1,len(text_white))]:
                        #         final_text.insert(len(final_text)-1, word)
                        #         second_finale = final_text
                                          
        else : 
            second_finale = text_white
    else :
        second_finale = final_text
    
    
    return ' '.join([r for r in second_finale if r != ''])


def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ksize = 3
    blured = cv2.GaussianBlur(gray, (ksize,ksize), 0)

    ret,threshed_b2b = cv2.threshold(blured,255,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    size = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size)) # or kernel = np.ones((size,size),np.uint8)
    morpho = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    # crop = cv2.morphologyEx(crop,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    ret,threshed_w2b = cv2.threshold(morpho,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    adapt_thresh_w2b = cv2.adaptiveThreshold(threshed_w2b,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,2)
    # threshed_w2b = cv2.GaussianBlur(threshed_w2b, (ksize,ksize), 0)
    # threshed_b2b = cv2.GaussianBlur(threshed_b2b, (ksize,ksize), 0)
    
    return threshed_b2b, threshed_w2b, adapt_thresh_w2b

def clean_text(text, words_dictionnary):
    exclude_list = ['@','#','²','§',"\\","|","{","}","[","]"]
    tmp_tab = [r for r in text.split(' ') if r != ""]
    text_tmp = ""
    for word in ' '.join(tmp_tab):
        if word not in exclude_list:
            text_tmp += ""+word
            
    text_list = list()
    map_word = dict()
    word_found = list()
    text_tmp = text_tmp.strip()
    
    for word in words_dictionnary.split(" "):
        word = word.lower()
        if word in text_tmp and word not in word_found:
            l = text_tmp.find(word)
            map_word[l] = word
            word_found.append(word)   
             
    if map_word :
        for (i,j) in sorted(map_word.items()):
            text_list.append(j)
            
    return ' '.join(text_list)

def get_ocr(image, config = "-l eng --oem 1 --psm 7",use_dict = False ):
    # config = ("-l eng --oem 1 --psm 7")
    words_dictionnary = "1 2 3 4 5 6 7 8 9 10 X Y WHEN CLICKED CLICK IF THEN CODE AND WE CAN'T DO IT WE'LL LEAD YOU IN THE RIGHT DIRECTION SAY JUST FILL OUT SIMPLE FORM BELOW YOU'LL BE DONE ASK CAN OR DEBUG WAIT JOIN ANSWER WERE ON WE'RE SPEAK YOU NEED WELCOME TO CAT'S CAT SHOP ALL THINGS THING HERE HELLO HELLO! FOR SECOND SECONDS PLAY SOUND SOUNDS  MEOW UNTIL SET MYVARIABLE MY VARIABLE PICK RANDOM PICKRANDOM FOREVER REPEAT MOVE ELSE STEP STEPS STEPS. MOUSE POINTER MOUSE-POINTER TOUCHING APPLE"
    if use_dict :
        with open('scratch_dictionnary.txt', 'r') as dic :
            print('***********',dic)
            words_dictionnary = dic.read()
            
    threshed_b2b, threshed_w2b, adapt_threshed_w2b = image_processing(image)
    
    # text = pytesseract.image_to_string(image,config=config,nice=2)
    # text_original = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    # text_original = clean_text(text_original,words_dictionnary)
    # print("without process => ",text_original)
    
    text = pytesseract.image_to_string(threshed_b2b,config=config,nice=2)
    text = text.replace("\n", ' ').strip()
    text_in_black = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    # text_in_black = clean_text(text_in_black,words_dictionnary)
    print("text_in_black => ",text_in_black)
    
    text = pytesseract.image_to_string(threshed_w2b,config=config,nice=2)
    text = text.replace("\n", ' ').strip()
    text_in_white = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    # text_in_white = clean_text(text_in_white,words_dictionnary)
    print("text_in_white => ",text_in_white)
    cv2.imshow("original ",image)
    cv2.imshow("threshed_b2b ",threshed_b2b)
    cv2.imshow("threshed_w2b ",threshed_w2b)
    cv2.waitKey(0)

    return get_final_text(text_in_black,text_in_white)
    # return final_text

