
import numpy as np
import base64
import io 
import os  
import pandas as pd
import seaborn as sns
from pathlib import Path, PurePath
from helper import getDataToCluster
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import manifold


dirpath = Path().parent.absolute()
#rawdatadir = PurePath.joinpath(dirpath,'data/gated/')
transformdatadir = PurePath.joinpath(dirpath,'data/transformed/')
diffdatadir = PurePath.joinpath(dirpath,'data/diff/')
gateddatadir = PurePath.joinpath(dirpath,'data/gated/')

def read_data(filename): 
    print("Reading file ", filename, "....")
    datafile = PurePath.joinpath(transformdatadir, filename)   
    df = pd.read_csv(datafile, header=None,skiprows = 1,)     
    return df

def convertImageToBase64(plot):
    pic_IObytes = io.BytesIO()
    plot.savefig(pic_IObytes,  format='png')
    pic_IObytes.seek(0)
    base64String = base64.b64encode(pic_IObytes.read())
    
    return base64String

def implementKMeans(data, noOfClusters, title):
    model = KMeans(n_clusters = noOfClusters)
    
    model = model.fit(data)

    plt.figure(figsize=(8,6))
    plt.scatter(data.iloc[:,0], data.iloc[:,1], marker='.', s=250, c=model.labels_.astype(np.float))
    plt.title('Clustering using ' + title)
#    plt.show    
    
    base64String = convertImageToBase64(plt)
    
    return base64String
   

#def applySVD1(data):
#    #Calculate SVD
#    U, sigma, V = np.linalg.svd(data)
#    
#    #To dataframe with two decimal places
#    v_df=pd.DataFrame(V)
#    v_df.apply(lambda x: np.round(x, decimals=2))
#    
#    #Lower value = More similar
#    plt.imshow(V, interpolation='none')
#    ax = plt.gca()
#    plt.xticks(range(len(v_df.columns.values)))
#    plt.yticks(range(len(v_df.index.values)))
#    ax.set_xticklabels(v_df.columns.values, rotation=90)
#    plt.colorbar();


def applySvd(data):
    #Calculate SVD
    u, s, v = np.linalg.svd(data, full_matrices=True)

    # Visualize variance    
#    var_explained = np.round(s**2/np.sum(s**2), decimals=3)
#    sns.barplot(x=list(range(1,len(var_explained)+1)),
#                y=var_explained, color="limegreen")

#    plt.xlabel('SVs', fontsize=16)
#    plt.ylabel('Percent Variance Explained', fontsize=16)
#    plt.savefig('svd_scree_plot.png',dpi=100)
    
    # Select top 2 
    svdDf = pd.DataFrame(u[:,0:2])
#    # Visualize on a scatter plot
#    sns.scatterplot(x=0, y=1, hue=0, data=svdDf, s=100, alpha=0.7)
#    plt.xlabel(' 1: {0}%'.format(var_explained[0]*100), fontsize=16)
#    plt.ylabel(' 2: {0}%'.format(var_explained[1]*100), fontsize=16)
    
    return svdDf
    

def applyMds(data):
    mds = manifold.MDS(n_components=2, dissimilarity="euclidean", random_state=5) #euclidean #Input is dissimalirity matrix if precomputed is used
    
    xyCoords = mds.fit(data)
#   
    #Positions of datasets in the embedding space
    mdsDf = pd.DataFrame(xyCoords.embedding_)
    
    return mdsDf

def applyMdsPrecomputed(data):
    mds = manifold.MDS(n_components=2, dissimilarity="precomputed") #Input is dissimalirity matrix if precomputed is used
    
    correlations = data.corr()
    plt.matshow(correlations)
    print(correlations)
    
    xyCoords = mds.fit(correlations.values)
#   
    #Positions of datasets in the embedding space
    mdsDf = pd.DataFrame(xyCoords.embedding_)
    
    return mdsDf

def loadClusterImages(size, noOfClusters):
    
    filelist  = os.listdir(gateddatadir)  
    filesToDf = getDataToCluster(filelist, size)
    fcsdata = pd.DataFrame.transpose(pd.DataFrame(filesToDf))
    
    svdData = applySvd(fcsdata)
    mdsData = applyMds(fcsdata)
    
    base64Img = {}
    
    base64Img['svd'] = (implementKMeans(svdData, noOfClusters, 'SVD')).decode('utf-8')
    base64Img['mds'] = (implementKMeans(mdsData, noOfClusters, 'MDS')).decode('utf-8')
    
    return base64Img


#plt.matshow(fcsdata.corr())

#
#
#filelist  = os.listdir(diffdatadir)  
#filesToDf = getDataToCluster(filelist, 9)
#fcsdata = pd.DataFrame.transpose(pd.DataFrame(filesToDf))
#
#
#y = applyMds(fcsdata)
#
#print(type(applyMds(fcsdata)))


loadClusterImages(81, 2)

