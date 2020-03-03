import cv2
from modules.imageProcessing import *
# OCR TRAITMENT


englishWordlist = ["move","steps","step","turn","degrees","degre","degree","go","to","point","direction","edge","bounce","bounc","set","rotation","style","if","change","random","position","when","clicked","clicke","click","space","up","arrow","down","left","right","any","key","pressed","presse","wait","second","second","repeat","repet","forever","then","else","until","stop","all","pick","random","and","or","not","mod","round","pen","apple","contains","contain"]
exceptList = ""#{[(\@&!?\"\'%#;|,.$^€¨ùàçaz€`_/)]}"

def setFilter(image,lang):
    filter_list = ["median","blur" ,"morpho","thresh","bilateral","gauss","2d"]  
    numbreOfLines = 0
    extractedText = ""
    bestFilter = ""
    worldsFound = 0
    image1 = image
    for key in filter_list:
        tmpImage = filter_list[key](image)
        extractedText = getText(tmpImage,lang)
        saveExtractedText(extractedText)
        dictionnaryLength = len(getDictionnaryText(extractedText))
        if dictionnaryLength > numbreOfLines:
            numbreOfLines = dictionnaryLength
            bestFilter = key
        elif dictionnaryLength == numbreOfLines:
            tempDictionnary = getDictionnaryText(extractedText)
            numbreOfWorlds = 0
            for line in tempDictionnary:
                for world in tempDictionnary[line].split(" "):
                    if world in englishWordlist:
                        numbreOfWorlds += 1
            if numbreOfWorlds > worldsFound:
                worldsFound = numbreOfWorlds
                numbreOfLines = dictionnaryLength
                bestFilter = key
    return image,getDictionnaryText(extractedText),bestFilter

def getText(image,lang):
    options = r'-l '+lang+' --oem 3 --psm 6'
    try:
        text = pytesseract.image_to_string(image,config=options,nice=2)
    except RuntimeError as timeout_error:
        pass
    return text
def saveExtractedText(text):
    with open("OCR_result/tmpExtractedText.txt","w") as myFile:
        myFile.write(text)        
def readExtractedText():
    with open("OCR_result/extractedText.txt","r") as myFile:
        for line in myFile:
            print(line)
def getDictionnaryText(text):
    tempDictionnary = dict()
    with open("OCR_result/tmpExtractedText.txt","r") as myFile:
        numbreOfLines = 0
        for line in myFile:
            line = line.replace('\n', '')
            line = line.split(" ")
            for word in line:
                if word in englishWordlist:
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
  