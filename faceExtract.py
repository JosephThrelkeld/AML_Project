from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pathlist = Path("att_faces")

def extractFaces(pathlist):
    facesTrain = np.zeros((320,10304))
    facesTest = np.zeros((80,10304))
    targetsTrain = np.zeros(320)
    targetsTest = np.zeros(80)
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
            if (picIDX > 7): #Add last two test data
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