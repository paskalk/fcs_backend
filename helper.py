
from differences import read_data
import json
import numpy as np

def getFcsFilesToUse(filestring, stringToAppend):
    fcsFiles = json.loads(filestring)
    filesList = []
    for item in fcsFiles:
        filesList.append(item["name"]+stringToAppend) 
    return filesList


def get3ColumnHeat(filename, xVal, yVal):

    df = read_data(filename).to_dict()  
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
    df = read_data(newFilename).to_dict()  
    
    gatedDict = {}
    counter1 = 0
    for key, val in df.items():
        dataList = []
        for item in val.values():
            dataList.append(item)
            gatedDict[counter1]=dataList
        counter1 = counter1 + 1
            
    return gatedDict

def _loadJsonGateData(filename, size):
    
#    newFilename = filename.split(".fcs")[0] + "-gate.csv"
    df = np.array(read_data(filename))
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


def getGatedLineData(fileList, x1,y1, x2, y2):
    size = (y2-y1)*(x2-x1)
    gatesDict = {}
    counter = 0
    for file in fileList:
        _gateDict = _loadJsonGateData(file, size)  
        gatesDict[file] = _gateDict
        counter = counter + 1
    
#    allFilesDict = {} 
#    allFilesDict["files"] = gatesDict
    return gatesDict


#df = getGatedLineData(["A02 Kranvatten kvall SYBR.fcs","A05 Kranvatten Sept SYBR .fcs","A02 Kranvatten Augusti SYBR.fcs","A03 Kranvatten Mars SYBR.fcs","A03 Kranvatten morgon SYBR.fcs"], 5,5,12,12)
#print(df)












