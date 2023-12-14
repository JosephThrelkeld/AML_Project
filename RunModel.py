from PerformPCA import *
from FaceExtract import *
from SupportVectorModel import *

from sklearn.metrics import classification_report
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler



val = int(input("Input the number of pictures per person to be used for testing (no more than 8), or press enter for default:") or "2")
#Extract faces and set training and test datasets
pathlist = Path("att_faces")
facesTrain, facesTest, targetsTrain, targetsTest = extractFaces(pathlist, val) 
facesTrainStandardized = (facesTrain - np.average(facesTrain,axis=0)) / np.std(facesTrain,axis=0)
facesTestStandardized = (facesTest - np.average(facesTest, axis=0)) / np.std(facesTest, axis=0)

#Components to be used for PCA reductions
nComp = 50

#Constructed Model
#--------------------------------------------------------------------------------------------------
#Perform PCA and reduce dimensionality of data 
print("Using constructed model...")
sortedEigenVecs = pca(facesTrain)
eigenVecSubset = sortedEigenVecs[:,0:nComp]
facesTrainReduced = np.dot(facesTrainStandardized,eigenVecSubset)
facesTestReduced = np.dot(facesTestStandardized,eigenVecSubset)

#Create and fit data to classifier
clfOwnModel = SVM_OVO(1000)
clfOwnModel.fit(facesTrainReduced,targetsTrain)

#Use classifier to predict new data 
predsTest = clfOwnModel.predict(facesTestReduced)
#--------------------------------------------------------------------------------------------------

#Skleran PCA SVM
#--------------------------------------------------------------------------------------------------
print("Using sklearn SVM model...")
#PCA
eigenVecSubset = RandomizedPCA(n_components=nComp,whiten=True).fit(facesTrain)
facesTrainReduced = np.dot(facesTrainStandardized,eigenVecSubset.components_.T)
facesTestReduced = np.dot(facesTestStandardized,eigenVecSubset.components_.T)

param_grid = {'C': [1e3,5e3,1e4,5e4,1e5]}
clfSKL_SVM = GridSearchCV(SVC(class_weight='balanced'),param_grid,cv=int(min(5,(len(facesTrain)/40))))
clfSKL_SVM.fit(facesTrainReduced,targetsTrain)
predsTestsklearnSVM = clfSKL_SVM.predict(facesTestReduced)
#--------------------------------------------------------------------------------------------------

#Sklearn MLPmodel
#--------------------------------------------------------------------------------------------------
print("Using sklearn MLP model...")
scaler = StandardScaler()
scaler.fit(facesTrain)
mlpFacesTrain = scaler.transform(facesTrain)
mlpFacesTest = scaler.transform(facesTest)
clfMLP = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(100,),random_state=42,max_iter=500)
clfMLP.fit(mlpFacesTrain,targetsTrain)

predsTestMLP = clfMLP.predict(mlpFacesTest)
#--------------------------------------------------------------------------------------------------

#Print out results
print("Classification report for constructed model:")
print(classification_report(targetsTest, predsTest))
print("Classification report for SVM model using sklearn:")
print(classification_report(targetsTest, predsTestsklearnSVM))
print("Classification report for Multi-layer perceptron model using sklearn:")
print(classification_report(targetsTest,predsTestMLP))