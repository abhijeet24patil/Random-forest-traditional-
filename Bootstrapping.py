from numpy import*
import random

def bootstrap(dataset,bNo):  ## bNo is Number of bootstrap sample
    sampledRecords=[]        ## list to store Single Bootstrap 
    bootstrapedData=[]       ## list to store bNo of bootstrap samples 
    for x in range(bNo):           
         sampledRecords=sampling(dataset)
         bootstrapedData.append(sampledRecords)
    return bootstrapedData   ## Bootstrap data of size bNo. 

def sampling(dataset):
    bootstrap=[]
    dataSize=len(dataset)
    for i in range(dataSize):               ##selects samples randomly l times 
         obs=random.choice(dataset)   
         bootstrap.append(obs)
    return bootstrap                 ##returns a bootstrap data
