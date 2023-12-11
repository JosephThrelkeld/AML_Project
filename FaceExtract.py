from PIL import Image
import numpy as np
from pathlib import Path

def extractFaces(pathlist, testSize):
    print("Extracting faces...")
    if (testSize > 9):
        raise Exception("Test size must be less or equal to 9")
    trainSize = 10 - testSize
    facesTrain = np.zeros((trainSize * 40,10304))
    facesTest = np.zeros((testSize * 40,10304))
    targetsTrain = np.zeros(trainSize * 40)
    targetsTest = np.zeros(testSize * 40)
    personIDX = 0
    trainIDX = 0
    testIDX = 0
    picIDX = 0
    for subDir in pathlist.iterdir():
        picIDX = 0
        for path in subDir.rglob('*.pgm'):
            pathStr = str(path)
            im = Image.open(pathStr)
            imArr= np.asarray(im)
            imArr = np.asarray(im).flatten()
            if (picIDX > trainSize - 1): #Add last two test data
                facesTest[testIDX] = imArr
                targetsTest[testIDX] = personIDX
                testIDX += 1
            else:
                facesTrain[trainIDX] = imArr
                targetsTrain[trainIDX] = personIDX
                trainIDX += 1
            picIDX += 1
        personIDX += 1

    return facesTrain, facesTest, targetsTrain, targetsTest 
