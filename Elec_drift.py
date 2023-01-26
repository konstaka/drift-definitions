from sklearn import preprocessing
import pandas as pd
import sklearn.svm as svm
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from scipy.io import arff
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import time
from sklearn.preprocessing import MinMaxScaler
from abc import ABC, abstractmethod
from sklearn import tree
from math import trunc
from math import ceil
from sklearn.neighbors import KNeighborsClassifier
#from c45 import C45
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

#sklearnex
#import sys
#sys.path.insert(0, "C:\Users\bapti\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearnex")
#import 
#import daal4py.sklearn


#regex
import re

#This is for optimization

#For randomness:
import random

#For different classifiers:
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



def createBestKNNDetector(data,labels,maxK):
    k_range = range(4, maxK)

    k_scores = np.zeros(maxK)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.30, shuffle=False)


    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        k_scores[k]= cross_val_score(model, data, labels, cv=10).mean()
        print(k_scores[k])

    optimizedNeighbors= k_scores.argmax()
    print("optimized neighbors for this is: " + str(optimizedNeighbors))
    return KNeighborsClassifier(n_neighbors=optimizedNeighbors)

#preprocessing.
#LabelEncoder
#iloc and sclice...
#functions to help make arf to usable data
def getCatColumns(pandaFrame):
    names = pandaFrame.select_dtypes(exclude=np.number).columns.tolist()
    return names
    
    #very inefficient, find way to get the last part and see if case, and match 
    #this is there to check what type of file it is and return a pandas frame.
def dfFromUnknownType(link):
    x = re.findall("arff$", link)

    if x:
        data = arff.loadarff(link)[0]
        return pd.DataFrame(data)
    else:
        x = re.findall("csv$",link)
        if x:
            return pd.read_csv(link)
        else:
            print("not a valid File Type, change code or filetype")

def divideIntoBatchesOfX(array,X):
    if array.ndim>1:
        return np.array_split(array,ceil(array.shape[0]/X))
    else:
        return np.array_split(array,ceil(array.size/X))

def getTestBatches(link, data, labels):
    changedFrame = np.delete(data[-30208:],np.array([0,1,7]),1)
    dat = divideIntoBatchesOfX(changedFrame,365)
    lab = divideIntoBatchesOfX(labels[-30208:],365)
    return dat,lab

def getTrainSet(link,data,labels):
	if (link =="electricity_dataset.csv"):
		return np.delete(data[:15104],np.array([0,1,7]),1),labels

    # match link:
    #     case "electricity_dataset.csv":
    #         A=np.delete(data[:15104],np.array([0,1,7]),1)
    #         return A,labels[:15104]
            #divideIntoBatchesOfX(data[:15104],365), divideIntoBatchesOfX(labels[:15104],365)

    

    # match link:
    #     case "electricity_dataset.csv":
    #         A=np.delete(data[-30208:],np.array([0,1,7]),1)
    #         return divideIntoBatchesOfX(A,365), divideIntoBatchesOfX(labels[-30208:],365)

    


def fromFiletoData(link):
    df= dfFromUnknownType(link)
    #encode labels
    le = preprocessing.LabelEncoder()
    labels= le.fit_transform(df[['label']])
    print("preprocessed",flush=True)
    #encode categorical
    columns = getCatColumns(df)
    print("getcolumns",flush=True)
    ord= preprocessing.OrdinalEncoder()
    scaler= MinMaxScaler()
    df[columns[:]]=ord.fit_transform(df[columns[:]])
    print("get the preprocessed things",flush=True)
    df[df.columns] = scaler.fit_transform(df[df.columns])
    df=df.drop(columns=['label'],axis=1)
    print(df)
    numpData=df.to_numpy()
    print("tonumpy.",flush=True)
    return numpData,labels

# link = "agraw1_1_abrupt_drift_0_noise_balanced.arff"
# data,labels = fromArftoData("agraw1_1_abrupt_drift_0_noise_balanced.arff")
# drifDet= DriftDetector(data[:30000],labels[:30000],"linear",1,10)
#print(data)
#print(labels)


def getClassifierAccuracy(link,data,labels,classifier):
    numberOfCrossValidations= 40
    trainBatches, totalTrainLabels = getTrainSet(link,data,labels)
    divBatch=divideIntoBatchesOfX(trainBatches,365)
    divLabels=divideIntoBatchesOfX(totalTrainLabels,365)
    accuracies = np.zeros(numberOfCrossValidations)
    for i in range(numberOfCrossValidations):
        rand=random.randint(0,len(divBatch)-1)
        trainData = np.concatenate(divBatch[:rand] + divBatch[rand+1:])
        trainLabels= np.concatenate(divLabels[:rand] + divLabels[rand+1:])
        classifier.fit(trainData,trainLabels)
        accuracies[i]=classifier.score(divBatch[rand],divLabels[rand])
    averageAccuracy= np.average(accuracies)
    standardDevAccuracy= np.std(accuracies)
    return averageAccuracy,standardDevAccuracy
    

realWorldSources= np.array([
    "electricity_dataset.csv",
    "weather_dataset.csv",
    "airline_dataset.csv",
    "spam_dataset.csv"])

# as the ammount of data goes to infinity, it will describe the thing the best. so the standard deviation will be... 
# statisticall, sampling a statistic stdDev
#This if for the drift calculations

def getClassifierAccuracy(link,data,labels,classifier):
    numberOfCrossValidations= 40
    trainBatches, totalTrainLabels = getTrainSet(link,data,labels)
    divBatch=divideIntoBatchesOfX(trainBatches,365)
    divLabels=divideIntoBatchesOfX(totalTrainLabels,365)
    accuracies = np.zeros(numberOfCrossValidations)
    for i in range(numberOfCrossValidations):
        rand=random.randint(0,len(divBatch)-1)
        trainData = np.concatenate(divBatch[:rand] + divBatch[rand+1:])
        trainLabels= np.concatenate(divLabels[:rand] + divLabels[rand+1:])
        classifier.fit(trainData,trainLabels)
        accuracies[i]=classifier.score(divBatch[rand],divLabels[rand])
    averageAccuracy= np.average(accuracies)
    standardDevAccuracy= np.std(accuracies)
    return averageAccuracy,standardDevAccuracy

#Outputs graph of each batch accuracy for each classifier
def findRealWorldDrift(theSources):
    
    ##this is getting the accuracies
    for ind in range(theSources.size):
        data,labels = fromFiletoData(theSources[ind])
        trainBatch,trainLabels= getTrainSet(theSources[ind],data,labels)
                ##classifiers to try stuff out.
        
        classifiers =np.array( [
        createBestKNNDetector(trainBatch,trainLabels,50),
        svm.SVC(kernel="linear", C=1),
        svm.SVC(kernel="poly", C=1),
        svm.SVC(gamma=2, C=1),
        #GaussianProcessClassifier(1.0 * RBF(1.0),  n_jobs = -1),
        DecisionTreeClassifier(max_depth=8),
        RandomForestClassifier(max_depth=8, n_estimators=11, max_features=2),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
            ])
        ############
        accuracies= np.zeros(classifiers.size)
        deviations= np.zeros(classifiers.size)
        ###finding expected accuracy and deviation, and fitting
        for classi in range(classifiers.size):
            accuracies[classi],deviations[classi] = getClassifierAccuracy(theSources[ind],data,labels,classifiers[classi])
            classifiers[classi].fit(trainBatch,trainLabels)
            
        ##This is finding when accuracies deviate
            testBatch,labelBatch= getTestBatches(theSources[ind],data,labels)
            batchAccuracies=np.zeros(ceil(len(testBatch)))
            drifts= np.zeros(len(testBatch))
            for batchNum in range(ceil(len(testBatch))):
                batchAccuracy = classifiers[classi].score(testBatch[batchNum],labelBatch[batchNum])
                batchAccuracies[batchNum]=batchAccuracy
                if accuracies[classi]-1*deviations[classi] > batchAccuracy:
                    drifts[batchNum]=True

            print(drifts)
            path = str(classi) + "path.csv"
            pd.DataFrame(drifts).to_csv(path)
            print(accuracies[classi])
            print(deviations[classi])
            plt.bar(np.arange(0,len(testBatch)),batchAccuracies)
            plt.axhline(accuracies[classi]-1*deviations[classi])
            plt.show()
    
findRealWorldDrift(realWorldSources)