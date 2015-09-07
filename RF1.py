##This is a code for traditional random forest. I have used the Decision tree code from book Machine Learning In action.
## I have made the requred changes to convert it into Random tree.

from numpy import*
import random
import time
from operator import*
from math import*
d=5

## random forest function
def randomforest(dataSet,q):      ##pass dataset and number of  bootstrap sample
     sub=[]
     cT=[]
     bStrap=bootstrap(dataSet,q)
     for s in bStrap:
          l0=s[0]
          l1=l0[:-1]
          crT=createTree(s,l1,d)
          cT.append(crT)
     return cT

## function for bootstrapping of dataset
def bootstrap(dat,q):
     #l=len(dat)
     bs=[]
     bbs=[]
     p=dat[0]
     for x in range(q):
          bs=sampl(dat)
          bbs.append(bs)
     for i in range(len(bbs)):
          bbs[i].insert(0,p)
     return bbs

def sampl(dat):
     w=[]
     l=len(dat)
     for i in range(l-1):
          oo=random.choice(dat[1:])
          w.append(oo)
          #ww=mat(w)
     return w

##function for random tree creation
def createTree(dataSet,labels,d):
     dat=dataSet[1:]
     classList=[ex[-1] for ex in dat]


     if classList.count(classList[0])==len(classList) or d==0:
          return classList[0]
     if len(labels)==1:
        return majorityCnt(classList)

     if len(dataSet[0])==0:
        return majorityCnt(classList)



##     if len(dat[0])>=2 and len(dat[0])<=4:
##        return majorityCnt(classList)
     l1=len(labels)
     m=int(sqrt(l1))
##     m=int(log(l1)+1)
     bestFeat=chooseBestFeatureToSplit1(dataSet,m)

     bestFeatLabel=labels[bestFeat]
     myTree={bestFeatLabel:{}}



     featValues=[ex[bestFeat]for ex in dat]
     uniqueVals=set(featValues)


     del(labels[bestFeat])
     for value in uniqueVals:
          subLabels=labels[:]
          data=splitDataSet(dataSet,bestFeat,value)

          myTree[bestFeatLabel][value]=createTree(data,subLabels,d-1)
     return myTree

def calcShannonEnt(dataSet):
     numentries=len(dataSet)
     labelCounts={}
     for feat in dataSet:
          currentLabel=feat[-1]
          if currentLabel not in labelCounts.keys():
               labelCounts[currentLabel]=1
          else:
               labelCounts[currentLabel]+=1
     shannonEnt=0.0
     for key in labelCounts:
          prob=float(labelCounts[key])/numentries
          shannonEnt-=prob*log(prob,2)
     return  shannonEnt

def splitDataSet(dataSet,axis,value):
     ff=dataSet[0].copy()
     gg=ff[axis]
     del(ff[axis])
     retDataSet=[]
     for feature in dataSet:
          if feature[axis]==value:
               reducedFeat=feature[:axis]
               reducedFeat.extend(feature[axis+1:])
               retDataSet.append(reducedFeat)
     retDataSet.insert(0,ff)
     return retDataSet

def chooseBestFeatureToSplit(dataSet):
     numFeatures=len(dataSet[0])-1
     baseEntropy=calcShannonEnt(dataSet)
     bestInfoGain=0.0;bestFeature=-1

     for i in range(numFeatures):
          featList=[example[i] for example in dataSet]
          uniqueVal=set(featList)

          newEntropy=0.0
          for value in uniqueVal:
               subDataSet=splitDataSet(dataSet,i,value)
               prob=len(subDataSet)/float(len(dataSet))
               newEntropy+=prob*calcShannonEnt(subDataSet)
          infoGain=baseEntropy-newEntropy

          if (infoGain>bestInfoGain):
               bestInfoGain=infoGain
               bestFeature=i
     return bestFeature

def createDataSet(fileName):
     dst=[]
     fr=open(fileName)
     for line in fr.readlines():
          curLine=line.strip().split('\t')
          rr=[]
          for word in curLine:
               rr=word.strip().split(',')
          dst.append(rr)
     return dst

def subDataSet(dataSet):
     x=len(dataSet[0])-1
     d1=[]
     for r in range(x):
          y=[e[r] for e in dataSet]
          d1.append(y)
     xx=mat(d1)
     yy=xx.T
     dataSet1=yy.tolist()
     ind=[]
     gg=[]
     f=dataSet1[0]
     s=random.sample(f,3)
     for i in s:
          p=dataSet1[0].index(i)
          ind.append(p)
     for j in ind:
          ah=[ex[j] for ex in dataSet1]
          gg.append(ah)
     m=mat(gg)
     mm=m.T
     li=mm.tolist()
     #subData=li[1:]
     #return subData
     f0=[ex[-1] for ex in dataSet]
     for r in range(len(li)):
          li[r].insert(len(li[r]),f0[r])
     return li

import operator
def majorityCnt(classList):
     classCount={}
     for vote in classList:
          if vote not in classCount.keys():
            classCount[vote]=1
          else:
            classCount[vote]+=1
     sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
     return sortedClassCount[0][0]



def splitDataSet1(dataSet,axis,value):
     retDataSet=[]
     for feature in dataSet:
          if feature[axis]==value:
               reducedFeat=feature[:axis]
               reducedFeat.extend(feature[axis+1:])
               retDataSet.append(reducedFeat)
     return retDataSet

def chooseBestFeatureToSplit1(dataSet,m):
     #numFeatures=len(dataSet[0])-1
     dat=dataSet[1:]
     baseEntropy=calcShannonEnt(dat)
     bestInfoGain=0.0;bestFeature=len(dat[0])-2

     ff=dataSet[0]
     f=ff[:-1]
     s=random.sample(f,m)
     ind=[]

     for i in s:
          p=f.index(i)
          ind.append(p)
     for i in ind:
          featList=[example[i] for example in dat]
          uniqueVal=set(featList)

          newEntropy=0.0
          for value in uniqueVal:
               subDataSet=splitDataSet1(dataSet,i,value)
               prob=len(subDataSet)/float(len(dat))
               newEntropy+=prob*calcShannonEnt(subDataSet)
          infoGain=baseEntropy-newEntropy
          if (infoGain>bestInfoGain):
               bestInfoGain=infoGain
               bestFeature=i
     return bestFeature


##function to classify new sample
def classify(inputTree,featLabels,testVec):
     firstStr = list(inputTree.keys())
     fs=firstStr[0]
     classLabel=0
     secondDict = inputTree[fs]
     featIndex = featLabels.index(fs)
     for key in secondDict.keys():
          if testVec[featIndex] == key:
               if type(secondDict[key]).__name__=='dict':
                    classLabel = classify(secondDict[key],featLabels,testVec)
               else:
                    classLabel = secondDict[key]
     return classLabel

def vote(featValue,trees,flabel):
     cl=[]
     dic={}
     for tree in trees:
          d=classify(tree,flabel,featValue)
          cl.append(d)
     for c in cl:
        if c in dic.keys():
            dic[c]+=1
        else:dic[c]=1

     sss=sorted(dic.items(),key=operator.itemgetter(1), reverse=True)
     return sss[0][0]


##for non-comma separeator
def createDataSet1(fileName):
     dst=[]
     fr=open(fileName)
     for line in fr.readlines():
          curLine=line.strip().split(' ')
          rr=[]
          for word in curLine:
               rr=word.strip().split('\t')
          dst.append(rr)
     return dst

