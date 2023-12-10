from PerformPCA import *

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.svm import SVC


pathlist = Path("att_faces")
facesTrain, facesTest, targetsTrain, targetsTest = extractFaces(pathlist) 
facesTrainStandardized = (facesTrain - np.average(facesTrain,axis=0)) / np.std(facesTrain,axis=0)
facesTestStandardized = (facesTest - np.average(facesTest, axis=0)) / np.std(facesTest, axis=0)


nComp = 50
sortedEigenVecs = pca(facesTrain)
eigenVecSubset = sortedEigenVecs[:,0:nComp]
eigenVecSubsetT = eigenVecSubset.T.real
#eigenVecSubset = RandomizedPCA(n_components=nComp,whiten=True).fit(facesTrain)


facesTrainReduced = np.dot(facesTrainStandardized,eigenVecSubset)
facesTestReduced = np.dot(facesTestStandardized,eigenVecSubset)



print(facesTrain.shape)
print(facesTrainReduced.shape)

print(facesTest.shape)
print(facesTestReduced.shape)

param_grid = {'C': [1e3,5e3,1e4,5e4,1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],}
clf = GridSearchCV(SVC(kernel='rbf',class_weight='balanced'),param_grid)
clf.fit(facesTrainReduced,targetsTrain)
print(clf.best_estimator_)

predsTest = clf.predict(facesTestReduced)

print(classification_report(targetsTest, predsTest))


#fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
#for i in range(16):
#    axes[i%4][i//4].imshow(eigenVecSubsetT[i].reshape(112,92),cmap='gray')
#plt.show()