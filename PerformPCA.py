import numpy as np

def pca(faces):
    print("   Performing PCA...")
    avgFace = np.average(faces,axis=0)
    X = faces
    X = X - np.average(X,axis=0)

    covMatrix = np.cov(X,rowvar = False)
    eigenVals, eigenVecs = np.linalg.eigh(covMatrix)
    sortedIDX = np.argsort(eigenVals)[::-1]
    sortedEigenVals = eigenVals[sortedIDX]
    sortedEigenVecs = eigenVecs[:, sortedIDX]
    
    return sortedEigenVecs
    
