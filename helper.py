
import json
import numpy as np
from pathlib import Path, PurePath
import pandas as pd 
import os

dirpath = Path().parent.absolute()

heatmapdatadir = PurePath.joinpath(dirpath,'data/heatmap/')
gateddatadir = PurePath.joinpath(dirpath,'data/gated/')
diffdatadir = PurePath.joinpath(dirpath,'data/diff/')

def read_data(filename, directorypath): 
    print("Reading file ", filename, "....")
    datafile = PurePath.joinpath(directorypath, filename)   #gateddata.csv'
    df = pd.read_csv(datafile, header=None)      #, index_col=0)
    return df

def clearGatedFiles():
    try:
        files = os.listdir(gateddatadir)
        for file in files:
            os.remove(PurePath.joinpath(gateddatadir, file))
            
        heatmapFiles = os.listdir(heatmapdatadir)
        for file in heatmapFiles:
            os.remove(PurePath.joinpath(heatmapdatadir, file))
            
    except Exception as e:
        print(e)
    

def clearDiffFiles():
    try:
        files = os.listdir(diffdatadir)
        for file in files:
            os.remove(PurePath.joinpath(diffdatadir, file))
    except Exception as e:
        print(e)

def getFcsFilesToUse(filestring, stringToAppend):
    fcsFiles = json.loads(filestring)
    filesList = []
    for item in fcsFiles:
        filesList.append(item["name"]+stringToAppend) 
    return filesList


def get3ColumnHeat(filename, xVal, yVal):

    df = read_data(filename, gateddatadir).to_dict()  
    gatedDict = {}
    counter1 = 0
    for key, val in df.items():
    
        for item in val.values():
            itemDict = {}
            itemDict['x'] = key
            itemDict['y'] = key
            itemDict["color"] = item
            
            gatedDict[counter1]=itemDict
            counter1 = counter1 + 1
            
    return gatedDict

def getGatedHeatData(filename):
    newFilename = filename.split(".fcs")[0] + "-gate.csv"
    df = read_data(newFilename, gateddatadir).to_dict()  
    
    gatedDict = {}
    counter1 = 0
    for key, val in df.items():
        dataList = []
        for item in val.values():
            dataList.append(item)
            gatedDict[counter1]=dataList
        counter1 = counter1 + 1
            
    return gatedDict

def _loadJsonGateData(filename, size, gatedfilespath):
    
#    newFilename = filename.split(".fcs")[0] + "-gate.csv"
    df = np.array(read_data(filename, gatedfilespath))
    data = np.reshape(df,size)
    
    intDict = {}
    counter = 0
    strList = list(data)
    for item in strList: 
        _dict = {}
        _dict['x'] = int(counter)
        _dict['y'] = int(item)
        intDict[counter] = _dict
        counter = counter + 1

    return intDict


def getGatedLineData(fileList, x1, y1, x2, y2):
    size = (y2-y1)*(x2-x1)
    gatesDict = {}
    counter = 0
    for file in fileList:
        _gateDict = _loadJsonGateData(file, size, gateddatadir)  
        gatesDict[file] = _gateDict
        counter = counter + 1
    
#    allFilesDict = {} 
#    allFilesDict["files"] = gatesDict
    return gatesDict



def _loadModelData(filename, size, gatedfilespath):
    
#    newFilename = filename.split(".fcs")[0] + "-gate.csv"
    df = np.array(read_data(filename, gatedfilespath))
    print(df.shape)
    data = np.reshape(df,size)
    
    intDict = {}
    counter = 0
    strList = list(data)
    for item in strList: 
        _dict = {}
        _dict['x'] = int(counter)
        _dict['y'] = int(item)
        intDict[counter] = _dict
        counter = counter + 1

    return data

def getDataToCluster(fileList, size):
    modelDict = {}
    counter = 0
    for file in fileList:
        _modelDict = _loadModelData(file, size, gateddatadir)  
        modelDict[file] = _modelDict
        counter = counter + 1

    return modelDict








'''
TEST STRUCTURE FOR SENDING TO HALI
'''
def _format(filename, size, gatedfilespath):
    
#    newFilename = filename.split(".fcs")[0] + "-gate.csv"
    df = np.array(read_data(filename, gatedfilespath))
    data = np.reshape(df,size)
    
    intDict = {}
    counter = 0
    strList = list(data)
    for item in strList: 
        _dict = {}
        _dict['x'] = int(counter)
        _dict['y'] = int(item)
        intDict[counter] = int(item)
        counter = counter + 1

    return intDict


def getData(fileList, x1, y1, x2, y2):
    size = (y2-y1)*(x2-x1)
    gatesDict = {}
    counter = 0
    for file in fileList:
        _gateDict = _format(file, size, gateddatadir)  
        gatesDict[file] = _gateDict
        counter = counter + 1
    
#    allFilesDict = {} 
#    allFilesDict["files"] = gatesDict
    return gatesDict

#df = getData(["A02 Kranvatten kvall SYBR-gate.csv","A05 Kranvatten Sept SYBR -gate.csv","A02 Kranvatten Augusti SYBR-gate.csv","A03 Kranvatten Mars SYBR-gate.csv","A03 Kranvatten morgon SYBR-gate.csv"], 5,5,12,12)
#print(df)












