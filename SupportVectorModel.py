import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel

class SVM_OVO:
    def __init__ (self,C):
        self.C = C
        self.classes = None 
        self.supportVecLabels = None
        self.classifiers = []
          
    def fit(self,samples,targets):
        print("   fitting data...")
        self.classes = np.unique(targets) #Setting classes for use in predict
        #Create classifier for each class pair
        for i in range(len(self.classes)):
            for j in range(i+1,len(self.classes)):
                c1, c2 = self.classes[i], self.classes[j]
                samplesPair,targetsPair = self.extractPair(samples,targets,c1,c2)
                
                classifier = SVC(C=self.C)
                classifier.fit(samplesPair,targetsPair)
                self.classifiers.append((c1,c2,classifier))
                
    def predictSample(self,sample):
        #Use all binary classifiers and use class with most votes
        votes = {}
        for c1,c2,classifier in self.classifiers:
            if (classifier.predict(sample.reshape(1,-1)) == 1):
                votes[c1] = votes.get(c1,0) + 1
            else:
                votes[c2] = votes.get(c2,0) + 1
        return max(votes, key=votes.get)

    def predict(self,samples):
        print("   prediciting labels for new data...")
        predictions = [self.predictSample(sample) for sample in samples]
        return np.array(predictions)
                
        
    def extractPair(self,samples,targets,c1,c2):
        #Extract only samples where labels are of either class
        #Make new labels(targets) to match if sample is of the first class
        mask = np.logical_or(targets == c1, targets == c2) 
        samplesPair = samples[mask]
        targetsPair = np.where(targets[mask] == c1, 1,-1)
        return samplesPair,targetsPair
        

    
        