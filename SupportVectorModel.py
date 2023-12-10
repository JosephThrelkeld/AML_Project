import numpy as np
import numexpr as ne
from sklearn.linear_model import SGDClassifier

class SVM_RBF_OVO:
    def __init__ (self,C,gamma):
        self.C = C
        self.gamma = gamma
        self.alpha = None
        self.classes = None 
        self.supportVecs = None
        self.supportVecLabels = None
        self.classifiers = []
        self.bias = None
        
    def rbfMatrix(self,samples):
        #K(x,x') = exp(-gamma||x-x'||^2)
        #||x-y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
        matrixNorm = np.sum(samples**2,axis=-1) #Summing squares of rows of matrix
        kernelMatrix = ne.evaluate('exp(-g*(A+B-2*C))', {
            'A': matrixNorm[:,None],
            'B': matrixNorm[None,:],
            'C': np.dot(samples,samples.T),
            'g': self.gamma
        })
        return kernelMatrix
    
    def fit(self,samples,targets):
        print("fitting data...")
        self.classes = np.unique(targets) #Setting classes for use in predict
        
        rbfSamples = self.rbfMatrix(samples)
        #Create classifier for each class pair
        for i in range(len(self.classes)):
            for j in range(i+1,len(self.classes)):
                c1, c2 = self.classes[i], self.classes[j]
                samplesPair,targetsPair = self.extractPair(rbfSamples,targets,c1,c2)
                
                classifier = SGDClassifier(loss="hinge",penalty="l2",max_iter=50)
                classifier.coef_ = self.C
                classifier.fit(samplesPair,targetsPair)
                self.classifiers.append((c1,c2,classifier))
                
    def predictSample(self,sample):
        print("prediciting labels for new data...")
        #Use all binary classifiers and use class with most votes
        votes = np.zeros(len(self.classes))
        for c1,c2,classifier in self.classifiers:
            if (classifier.predict(sample.reshape(1,-1)) == 1):
                votes[c1] += 1
            else:
                votes[c2] += 1
        return max(votes)

    def predict(self,samples):
        rbfSamples = self.rbfMatrix(samples)
        predictions = [self.predictSample(sample) for sample in rbfSamples]
        return np.array(predictions)
                
        
    def extractPair(self,samples,targets,c1,c2):
        #Extract only samples where labels are of either class
        #Make new labels(targets) to match if sample is of the first class
        mask = np.logical_or(targets == c1, targets == c2) 
        samplesPair = samples[mask]
        targetsPair = np.where(targets[mask] == c1, 1,-1)
        return samplesPair,targetsPair
        

    
        