from PerformPCA import *
from FaceExtract import *
from SupportVectorModel import *

from sklearn.metrics import classification_report
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.model_selection import GridSearchCV


val = input("Input the number of pictures per person to be used for testing (no more than 8):")

#Extract faces and set training and test datasets
pathlist = Path("att_faces")
facesTrain, facesTest, targetsTrain, targetsTest = extractFaces(pathlist, int(val)) 
facesTrainStandardized = (facesTrain - np.average(facesTrain,axis=0)) / np.std(facesTrain,axis=0)
facesTestStandardized = (facesTest - np.average(facesTest, axis=0)) / np.std(facesTest, axis=0)

#Constructed Model
#--------------------------------------------------------------------------------------------------
#Perform PCA and reduce dimensionality of data 
nComp = 50
sortedEigenVecs = pca(facesTrain)
eigenVecSubset = sortedEigenVecs[:,0:nComp]
eigenVecSubsetT = eigenVecSubset.T.real
facesTrainReduced = np.dot(facesTrainStandardized,eigenVecSubset)
facesTestReduced = np.dot(facesTestStandardized,eigenVecSubset)

#Create and fit data to classifier
clf = SVM_RBF_OVO(1000)
clf.fit(facesTrainReduced,targetsTrain)

#Use classifier to predict new data 
predsTest = clf.predict(facesTestReduced)
#--------------------------------------------------------------------------------------------------

#Skleran PCA SVM
#--------------------------------------------------------------------------------------------------
print("Using skleran model...")
#PCA
eigenVecSubset = RandomizedPCA(n_components=nComp,whiten=True).fit(facesTrain)
facesTrainReduced = np.dot(facesTrainStandardized,eigenVecSubset.components_.T)
facesTestReduced = np.dot(facesTestStandardized,eigenVecSubset.components_.T)

param_grid = {'C': [1e3,5e3,1e4,5e4,1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],}
clf = GridSearchCV(SVC(kernel='rbf',class_weight='balanced'),param_grid)
clf.fit(facesTrainReduced,targetsTrain)
predsTestsklearn = clf.predict(facesTestReduced)
#--------------------------------------------------------------------------------------------------

#

print("Classification report for constructed model:")
print(classification_report(targetsTest, predsTest))
print("Classification report for SVM model using sklearn:")
print(classification_report(targetsTest, predsTestsklearn))